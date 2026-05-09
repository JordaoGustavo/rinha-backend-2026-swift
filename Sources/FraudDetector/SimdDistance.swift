import CSimd

public enum SimdDistance {
    @inline(__always)
    public static func int16L2Squared(_ a: UnsafePointer<Int16>, _ b: UnsafePointer<Int16>) -> Int32 {
        Int32(csimd_int16_l2_squared(a, b))
    }
    @inline(__always)
    public static func int16L2SquaredFirst8(_ a: UnsafePointer<Int16>, _ b: UnsafePointer<Int16>) -> Int32 {
        Int32(csimd_int16_l2_squared_first8(a, b))
    }
    @inline(__always)
    public static func int16BboxLowerBound(_ query: UnsafePointer<Int16>, _ bboxMin: UnsafePointer<Int16>, _ bboxMax: UnsafePointer<Int16>) -> Int32 {
        Int32(csimd_int16_bbox_lower_bound(query, bboxMin, bboxMax))
    }
    @inline(__always)
    public static func float32L2Squared(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>) -> Float {
        csimd_float32_l2_squared(a, b)
    }
    @inline(__always)
    public static func prefetch(_ ptr: UnsafeRawPointer) {
        csimd_prefetch(ptr)
    }
}
