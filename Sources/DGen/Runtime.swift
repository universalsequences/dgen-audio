import Foundation

public protocol CompiledKernelRuntime {
    func run(outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int)
}

public class CCompiledKernel: CompiledKernelRuntime {
    public let source: String
    public let symbolName: String = "process"

    private var dylibHandle: UnsafeMutableRawPointer? = nil
    private var processFn: (@convention(c) (UnsafeMutablePointer<Float>, UnsafePointer<Float>, Int32) -> Void)?

    init(source: String) {
        self.source = source
    }

    public func compileAndLoad() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let cFile = tmpDir.appendingPathComponent("kernel.c")
        let oFile = tmpDir.appendingPathComponent("kernel.o")
        let dylibFile = tmpDir.appendingPathComponent("libkernel.dylib")

        try source.write(to: cFile, atomically: true, encoding: .utf8)

        let compile = Process()
        compile.launchPath = "/usr/bin/clang"
        compile.arguments = [
            "-O3", "-march=armv8-a", "-fPIC", "-shared",
            "-o", dylibFile.path, cFile.path
        ]
        compile.launch()
        compile.waitUntilExit()

        guard compile.terminationStatus == 0 else {
            throw NSError(domain: "CompileError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to compile kernel"])
        }

        dylibHandle = dlopen(dylibFile.path, RTLD_NOW)
        guard let handle = dylibHandle else {
            throw NSError(domain: "DLError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to load .dylib"])
        }

        guard let sym = dlsym(handle, symbolName) else {
            throw NSError(domain: "DLError", code: 3, userInfo: [NSLocalizedDescriptionKey: "Symbol \(symbolName) not found"])
        }

        processFn = unsafeBitCast(sym, to: (@convention(c) (UnsafeMutablePointer<Float>, UnsafePointer<Float>, Int32) -> Void).self)
    }

    public func run(outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int) {
        guard let fn = processFn else {
            fatalError("Kernel not compiled/loaded")
        }
        fn(outputs, inputs, Int32(frameCount))
    }
}
