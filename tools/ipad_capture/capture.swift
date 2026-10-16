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

// A quarter-scale greyscale of the whole frame, ~15KB as a JPEG. It carries
// what the text mask cannot: the face-down card backs beside each seat, which
// are how a folded player is told from a live one. Quality and scale are low
// on purpose -- the consumer only measures what fraction of a small box is
// mid-bright, which survives heavy compression.
let THUMB_SCALE = 0.25

func tableThumb(_ cg: CGImage) -> String? {
    let w = Int(Double(cg.width) * THUMB_SCALE), h = Int(Double(cg.height) * THUMB_SCALE)
    // Draw to RGBA and compute luma with the same weights OpenCV uses, rather
    // than rendering into a grey colourspace. CGColorSpace.linearGray applies
    // a linear transfer function and crushes the mid-tones this is measuring:
    // the same frame came out with a mean of 5.6 against OpenCV's 28.1, so
    // every brightness threshold on the consumer side missed.
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }
    ctx.interpolationQuality = .medium
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

    var grey = [UInt8](repeating: 0, count: w * h)
    for i in 0..<(w * h) {
        let r = Double(rgba[i * 4]), g = Double(rgba[i * 4 + 1]), b = Double(rgba[i * 4 + 2])
        grey[i] = UInt8(min(255.0, 0.299 * r + 0.587 * g + 0.114 * b))
    }
    guard let space = CGColorSpace(name: CGColorSpace.linearGray),
          let out = CGContext(data: &grey, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w, space: space,
                              bitmapInfo: CGImageAlphaInfo.none.rawValue),
          let small = out.makeImage() else { return nil }
    let rep = NSBitmapImageRep(cgImage: small)
    guard let jpeg = rep.representation(using: .jpeg,
                                        properties: [.compressionFactor: 0.6])
    else { return nil }
    return jpeg.base64EncodedString()
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
                     hero: String? = nil, thumb: String? = nil,
                     stages: [String: Double]? = nil) {
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
        if let stages = stages { payload["stage_ms"] = stages }
        if let mask = mask { payload["text_mask"] = mask }
        if let strip = strip {
            payload["board_strip"] = strip
            payload["board_strip_rect"] = [BOARD_RECT.x, BOARD_RECT.y,
                                           BOARD_RECT.w, BOARD_RECT.h]
        }
        if let thumb = thumb {
            payload["table_thumb"] = thumb
            payload["table_thumb_scale"] = THUMB_SCALE
        }
        if let hero = hero {
            payload["hero_strip"] = hero
            payload["hero_strip_rect"] = [HERO_RECT.x, HERO_RECT.y,
                                          HERO_RECT.w, HERO_RECT.h]
        }
        // When the line leaves the helper. Against timestamp_sec (capture)
        // this splits the lag into the helper's share and the consumer's.
        payload["emitted_sec"] = Date().timeIntervalSince1970
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
                     strip: boardStrip(cg), mask: textMask(cg), hero: heroStrip(cg),
                     thumb: tableThumb(cg))
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
    let ctx = CIContext()
    // Serial: sample-buffer callbacks and the held-frame flush both run here,
    // so neither needs a lock.
    let queue = DispatchQueue(label: "capture")

    init(fast: Bool, minInterval: Double) { self.fast = fast; self.minInterval = minInterval }

    /// Has the screen changed since the last read? A poker table is static
    /// most of the time -- players are thinking -- and re-reading an unchanged
    /// frame costs the same as reading a new one, so those are dropped before
    /// Vision runs.
    ///
    /// Counted in cells, not summed. The gate used to sum a 16x16 linear-gray
    /// thumbnail against a total of 256, and the things worth reading are
    /// small and bright on dark felt: an action bubble, a pile of chips, a
    /// stack going down. Replayed over 3 minutes at full rate, the frame where
    /// the previous player's bubble and chips finished landing -- 300-600ms
    /// into the hero's turn -- scored 44-80, so it was never read on any of 5
    /// turns, and the session produced 26 reads in all. Six cells of the
    /// 64x64 grid moving by 40 caught all 5 (110 reads).
    let CHANGE_DELTA = 40
    let CHANGE_CELLS = 6
    var lastGrid: [Int] = []

    private func changed(_ grid: [Int]) -> Bool {
        guard grid.count == lastGrid.count, !grid.isEmpty else { return true }
        var cells = 0
        for i in 0..<grid.count where abs(grid[i] - lastGrid[i]) >= CHANGE_DELTA {
            cells += 1
            if cells >= CHANGE_CELLS { return true }
        }
        return false
    }

    // The newest frame that arrived inside the throttle window, and when.
    // Dropping it outright lost the trailing edge: a bet landing 0.2s after
    // an emit, followed by a still screen, was never shown until something
    // else moved -- the device sends frames on change, so no later frame
    // comes along to carry it. It is held instead and read when the window
    // closes. Only touched on the delegate's serial queue.
    var pending: (px: CVPixelBuffer, at: Date)?
    var flushScheduled = false

    // Blink rejection. The client blanks a player's name/stack plate for
    // exactly one screen refresh at the instant they act -- measured live at
    // 37 fps over 3 minutes: two blanks, one frame each, back within 17-21ms.
    // The hero's turn begins at that same instant, and the content-diff gate
    // picked the blank out precisely because it is a big sudden change, so the
    // player who had just acted went missing from the first frame of the turn.
    //
    // Waiting for the screen to hold still does not work: the turn timer and
    // glows animate constantly, so a new frame arrives every ~18ms and nothing
    // is ever still. Instead each frame is read only once the NEXT one has
    // arrived, and rejected if some regions went dark for that one frame and
    // came straight back. Dark, not just different: chips and cards sliding
    // over the felt brighten a region for a frame, and counting those flagged
    // 158 of 6,749 frames; darkening alone flags 14 and still catches both
    // real blanks. Cost: one device frame (~18ms), or BLINK_WAIT on a still
    // screen, where the last frame cannot be a blank -- one is always followed
    // at once by the refresh that restores it.
    let BLINK_WAIT: TimeInterval = 0.035
    let BLINK_DROP = 25          // luma a region must lose, against both neighbours
    let BLINK_SAME = 12          // ...while its neighbours agree with each other
    let BLINK_CELLS = 3          // regions that must do it (the real blanks: 4 and 6)
    var arrivals = 0
    var before: [Int]?                                   // frame n-2
    var candidate: (px: CVPixelBuffer, at: Date, grid: [Int])?   // frame n-1

    /// 64x64 mean luma, from a 4x4 grid of samples per cell. Read straight
    /// from the BGRA buffer, since converting every frame would cost ~18ms.
    private func lumaGrid(_ px: CVPixelBuffer) -> [Int] {
        CVPixelBufferLockBaseAddress(px, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(px, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(px) else { return [] }
        let w = CVPixelBufferGetWidth(px), h = CVPixelBufferGetHeight(px)
        let row = CVPixelBufferGetBytesPerRow(px)
        let p = base.assumingMemoryBound(to: UInt8.self)
        let n = 64, k = 4
        var grid = [Int](repeating: 0, count: n * n)
        for cy in 0..<n {
            for cx in 0..<n {
                var sum = 0
                for sy in 0..<k {
                    let y = ((cy * k + sy) * 2 + 1) * h / (n * k * 2)
                    for sx in 0..<k {
                        let x = ((cx * k + sx) * 2 + 1) * w / (n * k * 2)
                        let o = y * row + x * 4                  // B, G, R, A
                        sum += (114 * Int(p[o]) + 587 * Int(p[o + 1])
                                + 299 * Int(p[o + 2])) / 1000
                    }
                }
                grid[cy * n + cx] = sum / (k * k)
            }
        }
        return grid
    }

    private func isBlink(_ a: [Int], _ b: [Int], _ c: [Int]) -> Bool {
        guard a.count == b.count, b.count == c.count, !a.isEmpty else { return false }
        var cells = 0
        for i in 0..<b.count where b[i] < a[i] - BLINK_DROP && b[i] < c[i] - BLINK_DROP
                                   && abs(a[i] - c[i]) < BLINK_SAME {
            cells += 1
        }
        return cells >= BLINK_CELLS
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput sb: CMSampleBuffer,
                       from c: AVCaptureConnection) {
        let arrived = Date()
        lastSample = arrived
        guard let px = CMSampleBufferGetImageBuffer(sb) else { return }
        if saveOnly {
            // Timing studies want every frame exactly as it arrived.
            process(px, capturedAt: arrived)
            return
        }
        let grid = lumaGrid(px)
        var next: [Int]? = nil
        if let held = candidate {
            // Without the frame before it there is nothing to judge a blink
            // against -- after a read, say, when frames were dropped -- so it
            // is passed over; this frame gets judged next instead.
            let reads = index
            if let prior = before {
                if !isBlink(prior, held.grid, grid) {
                    offer(held.px, at: held.at)
                } else if let dir = saveDir {
                    // Keep the evidence, so a rejection can be checked by eye.
                    note(String(format: "blink rejected %.0f", held.at.timeIntervalSince1970 * 1000))
                    let ci = CIImage(cvPixelBuffer: held.px)
                    if let cg = ctx.createCGImage(ci, from: ci.extent) {
                        saveThumb(cg, at: held.at, dir: dir + "/blinks")
                    }
                }
            }
            // A read takes ~250ms and drops whatever arrives meanwhile, so
            // the next arrival will not follow this one: judge it afresh.
            next = index == reads ? held.grid : nil
        }
        before = next
        candidate = (px, arrived, grid)
        arrivals += 1
        let generation = arrivals
        queue.asyncAfter(deadline: .now() + BLINK_WAIT) { [weak self] in
            guard let self = self, self.arrivals == generation,
                  let held = self.candidate else { return }
            // Nothing followed it: a still screen, and a blank never ends one.
            self.candidate = nil
            self.before = nil
            self.offer(held.px, at: held.at)
        }
    }

    // A settled frame, subject to the throttle.
    // A clean frame, subject to the settle delay and the throttle. With a
    // settle delay the frame read is not the first one after a change but
    // the newest one SETTLE later: bubbles, chips and badges animate in over
    // a few refreshes, and the first frame can catch them half-drawn.
    private func offer(_ px: CVPixelBuffer, at arrived: Date) {
        let wait = max(minInterval - Date().timeIntervalSince(lastEmit), settleDelay)
        if wait <= 0 {
            pending = nil
            process(px, capturedAt: arrived)
            return
        }
        // Keep only the newest; anything older is already superseded.
        pending = (px, arrived)
        if !flushScheduled {
            flushScheduled = true
            queue.asyncAfter(deadline: .now() + wait) { [weak self] in
                guard let self = self else { return }
                self.flushScheduled = false
                if let held = self.pending {
                    self.pending = nil
                    self.process(held.px, capturedAt: held.at)
                }
            }
        }
    }

    private func process(_ px: CVPixelBuffer, capturedAt: Date) {
        let t0 = Date()
        let ci = CIImage(cvPixelBuffer: px)
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return }
        if saveOnly, let dir = saveDir {
            // Timing study: every frame the device sends, no settling, no
            // Vision, no diff gate -- the gate would swallow a brief flash.
            saveThumb(cg, at: capturedAt, dir: dir)
            return
        }
        let grid = lumaGrid(px)
        guard changed(grid) else { return }
        lastGrid = grid
        lastEmit = Date()
        let t1 = Date()
        let obs = recogniseText(cg, fast: fast)
        let t2 = Date()
        let strip = boardStrip(cg), mask = textMask(cg), hero = heroStrip(cg),
            thumb = tableThumb(cg)
        let t3 = Date()
        // Where this frame's time went, so a slow one can be pinned on a
        // stage instead of guessed at.
        let ms = { (a: Date, b: Date) in (b.timeIntervalSince(a) * 1000).rounded() }
        let stages: [String: Double] = ["held": ms(capturedAt, t0), "convert": ms(t0, t1),
                                        "vision": ms(t1, t2), "encode": ms(t2, t3)]
        // The time the frame was captured, not the time it was read, so the
        // consumer can show how far behind the table it really is.
        Emitter.emit(index: index, timestamp: capturedAt.timeIntervalSince1970,
                     width: cg.width, height: cg.height, observations: obs,
                     strip: strip, mask: mask, hero: hero, thumb: thumb, stages: stages)
        if let dir = saveDir { saveFrame(cg, index: index, dir: dir) }
        index += 1
        // Frames arriving while this ran were dropped, so the history no
        // longer describes consecutive frames.
        before = nil
    }
}

// --save-dir: write each emitted frame to disk as <frame_index>.png, so what
// the reader produced can be checked against what was actually on screen.
// Diagnostic only. Encoding a full frame costs ~100ms, so it runs off the
// capture queue rather than adding to every frame's lag.
var saveDir: String?
let saveQueue = DispatchQueue(label: "save")

// --save-only (with --save-dir): skip recognition and save a quarter-scale
// JPEG of every frame the device sends, named by capture time in ms. For
// timing things on screen that are too brief for the ~4 fps the Vision path
// manages -- e.g. how long the client hides a plate after an action.
var saveOnly = false

// --settle-ms: how long after a change to wait before reading, taking the
// newest clean frame at that point. Default 30ms.
var settleDelay: TimeInterval = 0.030

func saveThumb(_ cg: CGImage, at: Date, dir: String) {
    saveQueue.async {
        let w = cg.width / 4, h = cg.height / 4
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return }
        ctx.interpolationQuality = .medium
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let small = ctx.makeImage() else { return }
        let rep = NSBitmapImageRep(cgImage: small)
        guard let jpg = rep.representation(using: .jpeg,
                                           properties: [.compressionFactor: 0.8])
        else { return }
        let name = String(format: "%.0f.jpg", at.timeIntervalSince1970 * 1000)
        try? jpg.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent(name)))
    }
}

func saveFrame(_ cg: CGImage, index: Int, dir: String) {
    saveQueue.async {
        let rep = NSBitmapImageRep(cgImage: cg)
        guard let png = rep.representation(using: .png, properties: [:]) else { return }
        let path = (dir as NSString).appendingPathComponent("\(index).png")
        try? png.write(to: URL(fileURLWithPath: path))
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
        output.setSampleBufferDelegate(grabber, queue: grabber.queue)
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
            // Report liveness every tick, whether or not anything was
            // emitted. The consumer sees only frames that got past the
            // content-diff gate, which a static table suppresses for as long
            // as the players are thinking -- so it cannot tell a quiet table
            // from a dead capture, and guessing from the gap between frames
            // marks a perfectly healthy table stale. Only this side knows,
            // because only this side sees arrivals before the gate.
            if let g = grabber {
                let quiet = g.lastSample == Date.distantPast
                    ? -1.0 : Date().timeIntervalSince(g.lastSample)
                note(String(format: "heartbeat %.1f", quiet))
            }
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

if let i = args.firstIndex(of: "--settle-ms"), i + 1 < args.count {
    settleDelay = (Double(args[i + 1]) ?? 30) / 1000
    args.removeSubrange(i...(i + 1))
}

if args.contains("--save-only") {
    saveOnly = true
    args.removeAll { $0 == "--save-only" }
}

if let i = args.firstIndex(of: "--save-dir"), i + 1 < args.count {
    saveDir = args[i + 1]
    try? FileManager.default.createDirectory(atPath: args[i + 1],
                                             withIntermediateDirectories: true)
    args.removeSubrange(i...(i + 1))
}

if let i = args.firstIndex(of: "--dir"), i + 1 < args.count {
    runDirectory(args[i + 1], fast: fast)
} else if args.contains("--live") {
    runLive(fast: fast, minInterval: minInterval)
} else {
    FileHandle.standardError.write("""
    usage: capture --dir <folder> [--fast]
           capture --live [--fast] [--min-interval <seconds>] [--settle-ms <ms>] [--save-dir <folder>]
    """.data(using: .utf8)!)
    exit(2)
}
