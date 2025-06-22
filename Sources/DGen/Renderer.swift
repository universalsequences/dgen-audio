// the beauty is this doesn't need to even know if its forward or backward

enum Device {
    case C
    case Metal
}

// ops for metal and C are C-style
// many operations will look the same
// with exception of SIMD intrinsics for C (wanna use mac accelerate simd operations)
// so renderer will need to, for C decide if its scalar, if so use same as metal
// otherwise use intrinsics
class CStyle {
    
}

// need a function for generating the scaffolding for C / metal (include statements if necessary)
//
//
// need a function for generating a given block
// for metal, this means a different kernel


class Renderer {
    private let device: Device;

    init(device: Device) {
        self.device = device
    }

    public func render(block: Block) {
        
    }
}


