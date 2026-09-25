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

// The hero's own cards, which the text mask cannot serve because suit is a
// colour. Covers the region on both devices: the fanned pair spans
// x 0.58-0.72 / y 0.78-0.86 on the iPad and x 0.58-0.83 / y 0.76-0.88 on the
// iPhone.
let HERO_RECT = (x: 0.55, y: 0.74, w: 0.30, h: 0.15)

func cropStrip(_ cg: CGImage, _ r: (x: Double, y: Double, w: Double, h: Double)) -> String? {
    let W = Double(cg.width), H = Double(cg.height)
    let rect = CGRect(x: r.x * W, y: r.y * H, width: r.w * W, height: r.h * H)
    guard let cropped = cg.cropping(to: rect) else { return nil }
    let rep = NSBitmapImageRep(cgImage: cropped)
    guard let png = rep.representation(using: .png, properties: [:]) else { return nil }
    return png.base64EncodedString()
}

func boardStrip(_ cg: CGImage) -> String? { cropStrip(cg, BOARD_RECT) }
func heroStrip(_ cg: CGImage) -> String? { cropStrip(cg, HERO_RECT) }

// MARK: - Text mask

// A full-resolution binary mask of the frame's text, white on black.
//
// Vision does not report one- and two-character stacks: it reads the handle
// and simply omits the number below it, which was 19 of the 29 stacks the
// fixed-region parser found and the Vision path missed. Recovering those
// needs pixels, and this is the cheapest way to carry them -- the mask is
// almost all background so it compresses to ~23KB, less than the board strip,
// while covering the whole frame at full resolution.
//
// The colour bands MUST stay in step with ocr_utils.name_stack_mask on the
// Python side: white for handles, warm orange for amounts. If those move,
// move these.
func textMask(_ cg: CGImage) -> String? {
    let w = cg.width, h = cg.height
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

    var gray = [UInt8](repeating: 0, count: w * h)
    for i in 0..<(w * h) {
        let r = Int(rgba[i * 4]), g = Int(rgba[i * 4 + 1]), b = Int(rgba[i * 4 + 2])
        let white  = r > 150 && g > 150 && b > 150
        let orange = r > 140 && g > 90 && b < 130 && (r - b) > 40
        gray[i] = (white || orange) ? 255 : 0
    }
    guard let space = CGColorSpace(name: CGColorSpace.linearGray),
          let out = CGContext(data: &gray, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w, space: space,
                              bitmapInfo: CGImageAlphaInfo.none.rawValue),
          let maskImage = out.makeImage() else { return nil }
    let rep = NSBitmapImageRep(cgImage: maskImage)
    guard let png = rep.representation(using: .png, properties: [:]) else { return nil }
    return png.base64EncodedString()
}

// MARK: - JSON output

struct Emitter {
    static func emit(index: Int, timestamp: Double, width: Int, height: Int,
                     observations: [(String, CGRect, Float)], source: String? = nil,
                     strip: String? = nil, mask: String? = nil,
                     hero: String? = nil) {
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
        if let mask = mask { payload["text_mask"] = mask }
        if let strip = strip {
            payload["board_strip"] = strip
            payload["board_strip_rect"] = [BOARD_RECT.x, BOARD_RECT.y,
                                           BOARD_RECT.w, BOARD_RECT.h]
        }
        if let hero = hero {
            payload["hero_strip"] = hero
            payload["hero_strip_rect"] = [HERO_RECT.x, HERO_RECT.y,
                                          HERO_RECT.w, HERO_RECT.h]
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
                     strip: boardStrip(cg), mask: textMask(cg), hero: heroStrip(cg))
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
    // When the iPad's screen locks, the device stays open and discoverable --
    // it simply stops delivering sample buffers. Tracked on ARRIVAL, before
    // any gating, because the content-diff gate legitimately suppresses
    // emission on a static table and that must not look like starvation.
    var lastSample = Date.distantPast
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
        lastSample = Date()
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
                     strip: boardStrip(cg), mask: textMask(cg), hero: heroStrip(cg))
        index += 1
    }
}

/// Whether an iOS device is on the USB bus at all.
///
/// AVFoundation cannot tell "nothing plugged in" apart from "plugged in but
/// not offering capture" -- both are an empty discovery list -- so the two are
/// separated here by asking the USB tree directly. The distinction matters
/// because the fixes are different: plug it in, versus unlock it and quit
/// QuickTime.
func iosDeviceOnBus() -> Bool {
    let ioreg = Process()
    ioreg.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
    ioreg.arguments = ["-p", "IOUSB", "-w0"]
    let pipe = Pipe()
    ioreg.standardOutput = pipe
    ioreg.standardError = FileHandle.nullDevice
    guard (try? ioreg.run()) != nil else { return false }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    ioreg.waitUntilExit()
    let text = String(data: data, encoding: .utf8) ?? ""
    return text.contains("iPad@") || text.contains("iPhone@")
}

func note(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

/// Keeps a capture session running across the device coming and going.
///
/// The capture interface disappears the moment the device locks -- iOS drops
/// it back to a USB configuration that doesn't publish one -- so a one-shot
/// startRunning() dies silently the first time the screen sleeps mid-session.
/// This polls instead: acquire when the device is there, notice when it goes,
/// and pick it up again when it returns.
// How long an open session may deliver nothing before we say so. Long enough
// not to trip on a brief stall, short enough to notice before a hand is over.
let STARVED_AFTER: TimeInterval = 5.0

final class Supervisor {
    let fast: Bool
    let minInterval: Double
    var session: AVCaptureSession?
    var grabber: LiveCapture?
    var lastComplaint = ""
    var frameIndex = 0

    init(fast: Bool, minInterval: Double) { self.fast = fast; self.minInterval = minInterval }

    private func complain(_ message: String) {
        // Only when the reason changes, so a long wait doesn't spam the log.
        if message != lastComplaint { note(message); lastComplaint = message }
    }

    private func start(_ device: AVCaptureDevice) {
        let session = AVCaptureSession()
        session.sessionPreset = .high
        guard let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            complain("waiting: device is present but held by another process "
                     + "(quit QuickTime Player)")
            return
        }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true      // latest frame wins
        // Carry the frame counter across reconnects so indices stay monotonic.
        let grabber = LiveCapture(fast: fast, minInterval: minInterval)
        grabber.index = frameIndex
        output.setSampleBufferDelegate(grabber, queue: DispatchQueue(label: "capture"))
        guard session.canAddOutput(output) else {
            complain("waiting: cannot add capture output")
            return
        }
        session.addOutput(output)
        session.startRunning()
        self.session = session
        self.grabber = grabber
        lastComplaint = ""
        note("capturing: \(device.localizedName)")
    }

    private func stop(_ reason: String) {
        frameIndex = grabber?.index ?? frameIndex
        session?.stopRunning()
        session = nil
        grabber = nil
        lastComplaint = ""
        note("lost the device (\(reason)) -- waiting for it to come back")
    }

    func tick() {
        let device = findDevice()
        if session != nil {
            if device == nil { stop("unplugged, or taken by another app") }
            else if let g = grabber,
                    Date().timeIntervalSince(g.lastSample) > STARVED_AFTER {
                // Open, present, and sending nothing. A locked screen looks
                // exactly like this: the device never disappears, it just
                // goes quiet.
                complain("no frames arriving -- unlock the iPad screen")
            } else if grabber?.lastSample != Date.distantPast {
                lastComplaint = ""
            }
            return
        }
        guard let device = device else {
            complain(iosDeviceOnBus()
                ? "waiting: iOS device is attached but offers no capture interface "
                  + "-- unlock it, and quit QuickTime Player if it is running"
                : "waiting: no iOS device attached over USB")
            return
        }
        start(device)
    }
}

func runLive(fast: Bool, minInterval: Double) -> Never {
    enableScreenCaptureDevices()
    let supervisor = Supervisor(fast: fast, minInterval: minInterval)
    supervisor.tick()
    Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in supervisor.tick() }
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
