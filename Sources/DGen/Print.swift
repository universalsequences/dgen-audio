public enum ANSI {
    public static let reset = "\u{001B}[0m"
    public static let bold = "\u{001B}[1m"

    public static let red = "\u{001B}[38;5;196m"  // bright red
    public static let green = "\u{001B}[38;5;46m"  // neon green
    public static let yellow = "\u{001B}[38;5;226m"  // bright yellow
    public static let blue = "\u{001B}[38;5;39m"  // electric blue
    public static let magenta = "\u{001B}[38;5;201m"  // hot pink
    public static let cyan = "\u{001B}[38;5;51m"  // aqua blue
    public static let orange = "\u{001B}[38;5;208m"  // deep orange
    public static let gray = "\u{001B}[38;5;245m"  // soft gray
    public static let white = "\u{001B}[38;5;15m"  // bright white
}

extension UOp {
    func prettyDescription() -> String {
        let opStr: String
        switch op {
        case let .add(a, b):
            opStr = "\(ANSI.green)add\(ANSI.reset)(\(a), \(b))"
        case let .sub(a, b):
            opStr = "\(ANSI.green)sub\(ANSI.reset)(\(a), \(b))"
        case let .mul(a, b):
            opStr = "\(ANSI.green)mul\(ANSI.reset)(\(a), \(b))"
        case let .div(a, b):
            opStr = "\(ANSI.green)div\(ANSI.reset)(\(a), \(b))"
        case let .abs(a):
            opStr = "\(ANSI.green)abs\(ANSI.reset)(\(a))"
        case let .sign(a):
            opStr = "\(ANSI.green)sign\(ANSI.reset)(\(a))"
        case let .floor(a):
            opStr = "\(ANSI.green)floor\(ANSI.reset)(\(a))"
        case let .ceil(a):
            opStr = "\(ANSI.green)ceil\(ANSI.reset)(\(a))"
        case let .round(a):
            opStr = "\(ANSI.green)round\(ANSI.reset)(\(a))"
        case let .memoryRead(base, offset):
            opStr = "\(ANSI.cyan)memoryRead\(ANSI.reset)(\(base), \(offset))"
        case let .memoryWrite(base, offset, value):
            opStr = "\(ANSI.cyan)memoryWrite\(ANSI.reset)(\(base), \(offset), \(value))"
        case let .delay1(a, b):
            opStr = "\(ANSI.cyan)delay1\(ANSI.reset)(\(a), \(b))"
        case let .sin(a):
            opStr = "\(ANSI.green)sin\(ANSI.reset)(\(a))"
        case let .cos(a):
            opStr = "\(ANSI.green)cos\(ANSI.reset)(\(a))"
        case let .tan(a):
            opStr = "\(ANSI.green)tan\(ANSI.reset)(\(a))"
        case let .tanh(a):
            opStr = "\(ANSI.green)tan\(ANSI.reset)(\(a))"
        case let .exp(a):
            opStr = "\(ANSI.green)exp\(ANSI.reset)(\(a))"
        case let .log(a):
            opStr = "\(ANSI.green)log\(ANSI.reset)(\(a))"
        case let .log10(a):
            opStr = "\(ANSI.green)log10\(ANSI.reset)(\(a))"
        case let .sqrt(a):
            opStr = "\(ANSI.green)sqrt\(ANSI.reset)(\(a))"
        case let .pow(base, exponent):
            opStr = "\(ANSI.green)pow\(ANSI.reset)(\(base), \(exponent))"
        case let .atan2(y, x):
            opStr = "\(ANSI.green)atan2\(ANSI.reset)(\(y), \(x))"
        case let .min(y, x):
            opStr = "\(ANSI.green)min\(ANSI.reset)(\(y), \(x))"
        case let .max(y, x):
            opStr = "\(ANSI.green)max\(ANSI.reset)(\(y), \(x))"
        case let .gt(a, b):
            opStr = "\(ANSI.yellow)gt\(ANSI.reset)(\(a), \(b))"
        case let .gte(a, b):
            opStr = "\(ANSI.yellow)gte\(ANSI.reset)(\(a), \(b))"
        case let .lte(a, b):
            opStr = "\(ANSI.yellow)lte\(ANSI.reset)(\(a), \(b))"
        case let .lt(a, b):
            opStr = "\(ANSI.yellow)lt\(ANSI.reset)(\(a), \(b))"
        case let .eq(a, b):
            opStr = "\(ANSI.yellow)eq\(ANSI.reset)(\(a), \(b))"
        case let .mod(a, b):
            opStr = "\(ANSI.yellow)mod\(ANSI.reset)(\(a), \(b))"
        case let .selector(_, _):
            opStr = "\(ANSI.yellow)selector\(ANSI.reset)()"
        case let .load(cell):
            opStr = "\(ANSI.cyan)load\(ANSI.reset)(\(cell))"
        case let .store(cell, val):
            opStr = "\(ANSI.cyan)store\(ANSI.reset)(\(cell), \(val))"
        case let .mutate(varID, val):
            opStr = "\(ANSI.red)mutate\(ANSI.reset)(\(varID), \(val))"
        case let .beginIf(cond):
            opStr = "\(ANSI.magenta)beginIf\(ANSI.reset)(\(cond))"
        case let .defineGlobal(cond):
            opStr = "\(ANSI.magenta)defineGlobal\(ANSI.reset)(\(cond))"
        case let .loadGlobal(cond):
            opStr = "\(ANSI.magenta)loadGlobal\(ANSI.reset)(\(cond))"
        case .endIf:
            opStr = "\(ANSI.magenta)endIf\(ANSI.reset)"
        case let .gswitch(a, b, t):
            opStr = "\(ANSI.green)switch\(ANSI.reset)(\(a), \(b), \(t))"
        case let .latch(a, b):
            opStr = "\(ANSI.green)latch\(ANSI.reset)(\(a), \(b))"
        case let .beginLoop(a, b):
            opStr = "\(ANSI.green)begin_loop\(ANSI.reset)(\(a), \(b))"
        case let .defineConstant(constantId, value):
            opStr = "\(ANSI.green)defineConstant\(ANSI.reset)(\(constantId),\(value))"
        case let .defineMemory(length):
            opStr = "\(ANSI.green)defineMemory\(ANSI.reset)(\(length))"
        case .endLoop:
            opStr = "\(ANSI.green)endLoop\(ANSI.reset)"
        case let .beginRange(a, b):
            opStr = "\(ANSI.green)beginRange\(ANSI.reset)(\(a), \(b))"
        case .endRange:
            opStr = "\(ANSI.green)endRange\(ANSI.reset)"
        case let .output(a, b):
            opStr = "\(ANSI.green)output\(ANSI.reset)(\(a), \(b))"
        case let .input(a):
            opStr = "\(ANSI.green)input\(ANSI.reset)(\(a))"
        case .frameCount:
            opStr = "\(ANSI.magenta)frameCount\(ANSI.reset)"
        case let .loadGrad(a):
            opStr = "\(ANSI.magenta)loadGrad\(ANSI.reset)(\(a))"
        case let .storeGrad(a, b):
            opStr = "\(ANSI.magenta)storeGrad\(ANSI.reset)(\(a), \(b))"
        case .frameIndex:
            opStr = "\(ANSI.magenta)frameIndex\(ANSI.reset)"
        }

        return "\(ANSI.bold)UOp\(ANSI.reset)(op: \(opStr), value: \(value))"
    }
}
