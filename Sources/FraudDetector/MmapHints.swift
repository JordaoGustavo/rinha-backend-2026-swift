#if canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#elseif canImport(Darwin)
import Darwin
#endif

public enum MmapHints {
    #if os(Linux)
    private static let MADV_HUGEPAGE: Int32 = 14
    private static let MADV_WILLNEED: Int32 = 3
    private static let MADV_RANDOM: Int32 = 1

    public static func hintHugePages(_ addr: UnsafeMutableRawPointer, _ length: Int) {
        _ = madvise(addr, length, MADV_HUGEPAGE)
    }
    public static func hintWillNeed(_ addr: UnsafeMutableRawPointer, _ length: Int) {
        _ = madvise(addr, length, MADV_WILLNEED)
    }
    public static func hintRandom(_ addr: UnsafeMutableRawPointer, _ length: Int) {
        _ = madvise(addr, length, MADV_RANDOM)
    }
    #else
    public static func hintHugePages(_ addr: UnsafeMutableRawPointer, _ length: Int) {}
    public static func hintWillNeed(_ addr: UnsafeMutableRawPointer, _ length: Int) {}
    public static func hintRandom(_ addr: UnsafeMutableRawPointer, _ length: Int) {}
    #endif
}
