import Testing
@testable import FraudDetector

@Test func int16L2SquaredBasic() {
    var a: [Int16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0]
    var b: [Int16] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0]

    let result = a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            SimdDistance.int16L2Squared(aPtr.baseAddress!, bPtr.baseAddress!)
        }
    }
    // Each dimension has diff=1, so sum = 14 * 1 = 14
    #expect(result == 14)
}

@Test func float32L2SquaredBasic() {
    var a: [Float] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var b: [Float] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    let result = a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            SimdDistance.float32L2Squared(aPtr.baseAddress!, bPtr.baseAddress!)
        }
    }
    #expect(result == 1.0)
}

@Test func bboxLowerBoundInside() {
    var query: [Int16] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0]
    var bmin:  [Int16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var bmax:  [Int16] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0]

    let result = query.withUnsafeBufferPointer { qPtr in
        bmin.withUnsafeBufferPointer { minPtr in
            bmax.withUnsafeBufferPointer { maxPtr in
                SimdDistance.int16BboxLowerBound(qPtr.baseAddress!, minPtr.baseAddress!, maxPtr.baseAddress!)
            }
        }
    }
    // Query is inside bbox, so lower bound = 0
    #expect(result == 0)
}

@Test func bboxLowerBoundOutside() {
    var query: [Int16] = [15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var bmin:  [Int16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var bmax:  [Int16] = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    let result = query.withUnsafeBufferPointer { qPtr in
        bmin.withUnsafeBufferPointer { minPtr in
            bmax.withUnsafeBufferPointer { maxPtr in
                SimdDistance.int16BboxLowerBound(qPtr.baseAddress!, minPtr.baseAddress!, maxPtr.baseAddress!)
            }
        }
    }
    // gap at dim 0 = 15 - 10 = 5, lb = 5*5 = 25
    #expect(result == 25)
}
