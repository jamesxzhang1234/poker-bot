// Capture + OCR helper for the iOS-device pipeline.
//
// Two modes, both emitting the same JSON-lines format on stdout so the Python
// side cannot tell them apart:
//
//   capture --dir <folder>   run over saved frames (backtests, calibration)
//   capture --live           capture the attached iOS device continuously
//
// Why this exists in Swift rather than Python: the device only appears to
// AVFoundation once a client sets the CoreMediaIO property below, and Apple's
// Vision framework reads a whole frame in ~90ms where a per-crop Tesseract
// cascade takes 8-11s. Doing capture and OCR in one process means Python
// receives located text instead of pixels.
import Foundation
import AVFoundation
import CoreMediaIO
import Vision
import AppKit

// MARK: - Board strip

// Card ranks are template-matched on the Python side, which needs pixels --
// but everything else needs only text, and shipping the whole 4MB frame per
// capture would throw away the main benefit of doing OCR here. So one small
// band is attached instead.
//
// These fractions cover the board on both layouts: the vertical band is
// almost identical across devices (0.417-0.483 of frame height on the iPad's
// 1640x2360 and 0.418-0.483 on the iPhone's 1206x2622), and the horizontal
// span is the wider of the two. The rect travels with the payload so the
// consumer maps slots from it rather than keeping a second copy of these
// numbers.
let BOARD_RECT = (x: 0.18, y: 0.40, w: 0.63, h: 0.10)

func boardStrip(_ cg: CGImage) -> String? {
    let W = Double(cg.width), H = Double(cg.height)
    let rect = CGRect(x: BOARD_RECT.x * W, y: BOARD_RECT.y * H,
                      width: BOARD_RECT.w * W, height: BOARD_RECT.h * H)
    guard let cropped = cg.cropping(to: rect) else { return nil }
    let rep = NSBitmapImageRep(cgImage: cropped)
    guard let png = rep.representation(using: .png, properties: [:]) else { return nil }
    return png.base64EncodedString()
}

// MARK: - JSON output

struct Emitter {
    static func emit(index: Int, timestamp: Double, width: Int, height: Int,
                     observations: [(String, CGRect, Float)], source: String? = nil,
                     strip: String? = nil) {
        var rows: [[String: Any]] = []
        let W = Double(width), H = Double(height)
        for (text, bb, conf) in observations {
            rows.append([
                "text": text,
                // Vision reports a normalised, bottom-left-origin box; the rest
                // of the pipeline thinks in top-left pixel coordinates.
                "x": Int(bb.midX * W),
                "y": Int((1 - bb.midY) * H),
                "w": Int(bb.width * W),
                "h": Int(bb.height * H),
                "confidence": Double(conf),
            ])
        }
        var payload: [String: Any] = [
            "frame_index": index, "timestamp_sec": timestamp,
            "width": width, "height": height, "observations": rows,
        ]
        // Directory mode names its source file. Consumers must key on this
        // rather than on line order: a file that fails to decode is skipped,
        // and anything zipping payloads against a directory listing by
        // position would silently pair every later frame with the wrong one.
        if let source = source { payload["source"] = source }
        if let strip = strip {
            payload["board_strip"] = strip
            payload["board_strip_rect"] = [BOARD_RECT.x, BOARD_RECT.y,
                                           BOARD_RECT.w, BOARD_RECT.h]
        }
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let line = String(data: data, encoding: .utf8) else { return }
        print(line)
        fflush(stdout)
    }
}

// MARK: - Vision

func recogniseText(_ cg: CGImage, fast: Bool) -> [(String, CGRect, Float)] {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = fast ? .fast : .accurate
    // Handles and chip amounts are not English words; correction invents them.
    req.usesLanguageCorrection = false
    req.recognitionLanguages = ["en-US"]
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    var out: [(String, CGRect, Float)] = []
    for obs in (req.results ?? []) {
        if let c = obs.topCandidates(1).first {
            out.append((c.string, obs.boundingBox, c.confidence))
        }
    }
    return out
}

// MARK: - Directory mode

func runDirectory(_ dir: String, fast: Bool) {
    let names = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? [])
        .filter { let l = $0.lowercased()
                  return l.hasSuffix(".png") || l.hasSuffix(".jpg") || l.hasSuffix(".jpeg") }
        .sorted()
    for (i, name) in names.enumerated() {
        let path = (dir as NSString).appendingPathComponent(name)
        guard let img = NSImage(contentsOfFile: path),
              let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            FileHandle.standardError.write("[\(i+1)/\(names.count)] \(name) SKIPPED (cannot decode)\n"
                                            .data(using: .utf8)!)
            continue
        }
        let t = Date()
        let obs = recogniseText(cg, fast: fast)
        FileHandle.standardError.write(
            "[\(i+1)/\(names.count)] \(name) \(cg.width)x\(cg.height) "
            .data(using: .utf8)!)
        FileHandle.standardError.write(
            String(format: "%.0fms %d obs\n", Date().timeIntervalSince(t)*1000, obs.count)
            .data(using: .utf8)!)
        Emitter.emit(index: i, timestamp: Date().timeIntervalSince1970,
                     width: cg.width, height: cg.height, observations: obs, source: name,
                     strip: boardStrip(cg))
    }
}

// MARK: - Live mode

func enableScreenCaptureDevices() {
    var prop = CMIOObjectPropertyAddress(
        mSelector: CMIOObjectPropertySelector(kCMIOHardwarePropertyAllowScreenCaptureDevices),
        mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
        mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))
    var allow: UInt32 = 1
    CMIOObjectSetPropertyData(CMIOObjectID(kCMIOObjectSystemObject), &prop, 0, nil,
                              UInt32(MemoryLayout<UInt32>.size), &allow)
}

func findDevice() -> AVCaptureDevice? {
    var types: [AVCaptureDevice.DeviceType] = []
    if #available(macOS 14.0, *) { types = [.external] } else { types = [.externalUnknown] }
    return AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .muxed,
                                            position: .unspecified).devices.first
}

final class LiveCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let fast: Bool
    let minInterval: Double
    var index = 0
    var lastEmit = Date.distantPast
    var lastSignature: [UInt8] = []
    let ctx = CIContext()

    init(fast: Bool, minInterval: Double) { self.fast = fast; self.minInterval = minInterval }

    /// Coarse 16x16 luma thumbnail. A poker table is static most of the time --
    /// players are thinking -- and re-reading an unchanged frame costs the same
    /// as reading a new one, so identical frames are dropped before Vision runs.
    private func signature(_ cg: CGImage) -> [UInt8] {
        var buf = [UInt8](repeating: 0, count: 16 * 16)
        guard let space = CGColorSpace(name: CGColorSpace.linearGray),
              let c = CGContext(data: &buf, width: 16, height: 16, bitsPerComponent: 8,
                                bytesPerRow: 16, space: space,
                                bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return [] }
        c.draw(cg, in: CGRect(x: 0, y: 0, width: 16, height: 16))
        return buf
    }

    private func changed(_ sig: [UInt8]) -> Bool {
        guard sig.count == lastSignature.count, !sig.isEmpty else { return true }
        var diff = 0
        for i in 0..<sig.count { diff += abs(Int(sig[i]) - Int(lastSignature[i])) }
        return diff > 256          // ~1 luma level per cell, averaged
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput sb: CMSampleBuffer,
                       from c: AVCaptureConnection) {
        guard Date().timeIntervalSince(lastEmit) >= minInterval,
              let px = CMSampleBufferGetImageBuffer(sb) else { return }
        let ci = CIImage(cvPixelBuffer: px)
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return }
        let sig = signature(cg)
        guard changed(sig) else { return }
        lastSignature = sig
        lastEmit = Date()
        let obs = recogniseText(cg, fast: fast)
        Emitter.emit(index: index, timestamp: Date().timeIntervalSince1970,
                     width: cg.width, height: cg.height, observations: obs,
                     strip: boardStrip(cg))
        index += 1
    }
}

func runLive(fast: Bool, minInterval: Double) -> Never {
    enableScreenCaptureDevices()
    Thread.sleep(forTimeInterval: 1.0)
    guard let dev = findDevice() else {
        // Distinguish the two silent failures: a device that is absent, and a
        // device that is present but held by QuickTime or unlocked-gated. Both
        // look identical to AVFoundation -- an empty discovery list.
        let ioreg = Process()
        ioreg.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
        ioreg.arguments = ["-p", "IOUSB", "-w0"]
        let pipe = Pipe(); ioreg.standardOutput = pipe
        try? ioreg.run(); ioreg.waitUntilExit()
        let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                          encoding: .utf8) ?? ""
        let attached = text.contains("iPad@") || text.contains("iPhone@")
        FileHandle.standardError.write((attached
            ? "error: iOS device is attached but exposes no capture interface.\n"
              + "       Unlock the device, and quit QuickTime Player if it is running.\n"
            : "error: no iOS device attached over USB.\n").data(using: .utf8)!)
        exit(1)
    }
    FileHandle.standardError.write("capturing: \(dev.localizedName)\n".data(using: .utf8)!)

    let session = AVCaptureSession()
    session.sessionPreset = .high
    guard let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
        FileHandle.standardError.write("error: cannot open device (held elsewhere?)\n"
                                        .data(using: .utf8)!)
        exit(1)
    }
    session.addInput(input)
    let out = AVCaptureVideoDataOutput()
    out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    out.alwaysDiscardsLateVideoFrames = true      // latest frame wins
    let delegate = LiveCapture(fast: fast, minInterval: minInterval)
    out.setSampleBufferDelegate(delegate, queue: DispatchQueue(label: "capture"))
    guard session.canAddOutput(out) else {
        FileHandle.standardError.write("error: cannot add output\n".data(using: .utf8)!)
        exit(1)
    }
    session.addOutput(out)
    session.startRunning()
    RunLoop.main.run()
    exit(0)
}

// MARK: - Entry

var args = Array(CommandLine.arguments.dropFirst())
let fast = args.contains("--fast")
args.removeAll { $0 == "--fast" }
var minInterval = 0.0
if let i = args.firstIndex(of: "--min-interval"), i + 1 < args.count {
    minInterval = Double(args[i + 1]) ?? 0.0
    args.removeSubrange(i...(i + 1))
}

if let i = args.firstIndex(of: "--dir"), i + 1 < args.count {
    runDirectory(args[i + 1], fast: fast)
} else if args.contains("--live") {
    runLive(fast: fast, minInterval: minInterval)
} else {
    FileHandle.standardError.write("""
    usage: capture --dir <folder> [--fast]
           capture --live [--fast] [--min-interval <seconds>]
    """.data(using: .utf8)!)
    exit(2)
}
