import Foundation

public enum TransactionParser {
    private static let fnvOffsetBasis: UInt64 = 14695981039346656037
    private static let fnvPrime: UInt64 = 1099511628211

    @inline(__always)
    private static func fnv1a(_ bytes: UnsafePointer<UInt8>, _ len: Int) -> UInt64 {
        var hash = fnvOffsetBasis
        for i in 0..<len {
            hash ^= UInt64(bytes[i])
            hash &*= fnvPrime
        }
        return hash
    }

    @inline(__always)
    private static func round4(_ value: Double) -> Float {
        Float((value * 10000).rounded() / 10000)
    }

    @inline(__always)
    private static func clamp01(_ value: Double) -> Double {
        min(max(value, 0), 1)
    }

    @inline(__always)
    private static func dayOfWeekMonBased(year: Int, month: Int, day: Int) -> Int {
        var y = year
        var m = month
        if m < 3 {
            m += 12
            y -= 1
        }
        let k = y % 100
        let j = y / 100
        var h = (day + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 - 2 * j) % 7
        if h < 0 { h += 7 }
        return (h + 5) % 7
    }

    public static func parse(_ json: UnsafeRawBufferPointer, into vector: UnsafeMutablePointer<Float>) {
        for i in 0..<16 { vector[i] = 0 }
        guard let baseAddr = json.baseAddress, json.count > 0 else { return }
        let bytes = baseAddr.assumingMemoryBound(to: UInt8.self)
        let len = json.count

        var txAmount: Double = 0
        var txInstallments: Double = 0
        var txYear = 0, txMonth = 0, txDay = 0, txHour = 0, txMinute = 0, txSecond = 0

        var custAvgAmount: Double = 0
        var custTxCount24h: Double = 0

        var knownHashes = (
            UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
            UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
            UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
            UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0)
        )
        var knownCount = 0
        var merchantIdHash: UInt64 = 0
        var hasMerchantId = false

        var merchantMcc = -1
        var merchantAvgAmount: Double = 0

        var isOnline = false
        var cardPresent = false
        var kmFromHome: Double = 0

        var hasLastTx = false
        var lastYear = 0, lastMonth = 0, lastDay = 0, lastHour = 0, lastMinute = 0, lastSecond = 0
        var lastKmFromCurrent: Double = 0

        // ctx: 0=root, 1=transaction, 2=customer, 3=merchant, 4=terminal, 5=last_transaction
        var ctx = 0
        var fieldId = 0
        var inKnownMerchants = false
        var rootFieldId = 0
        var pos = 0

        while pos < len {
            let ch = bytes[pos]
            if ch <= 0x20 { pos += 1; continue }

            switch ch {
            case 0x7B: // {
                if ctx == 0 {
                    switch fieldId {
                    case 10: ctx = 1
                    case 11: ctx = 2
                    case 12: ctx = 3
                    case 13: ctx = 4
                    case 14: ctx = 5; hasLastTx = true
                    default: break
                    }
                }
                fieldId = 0
                pos += 1

            case 0x7D: // }
                if ctx != 0 { ctx = 0 }
                pos += 1

            case 0x5B: // [
                if ctx == 2 && fieldId == 6 {
                    inKnownMerchants = true
                    fieldId = 0
                }
                pos += 1

            case 0x5D: // ]
                inKnownMerchants = false
                pos += 1

            case 0x22: // "
                pos += 1
                let strStart = pos
                while pos < len && bytes[pos] != 0x22 { pos += 1 }
                let strEnd = pos
                pos += 1

                var nextPos = pos
                while nextPos < len && bytes[nextPos] <= 0x20 { nextPos += 1 }

                if nextPos < len && bytes[nextPos] == 0x3A { // :  → property name
                    pos = nextPos + 1
                    fieldId = 0
                    let sLen = strEnd - strStart
                    if ctx == 0 {
                        if sLen == 11 && matchBytes(bytes + strStart, "transaction", 11)      { fieldId = 10; rootFieldId = 10 }
                        else if sLen == 8 && matchBytes(bytes + strStart, "customer", 8)       { fieldId = 11; rootFieldId = 11 }
                        else if sLen == 8 && matchBytes(bytes + strStart, "merchant", 8)       { fieldId = 12; rootFieldId = 12 }
                        else if sLen == 8 && matchBytes(bytes + strStart, "terminal", 8)       { fieldId = 13; rootFieldId = 13 }
                        else if sLen == 16 && matchBytes(bytes + strStart, "last_transaction", 16) { fieldId = 14; rootFieldId = 14 }
                    } else if ctx == 1 {
                        if sLen == 6 && matchBytes(bytes + strStart, "amount", 6)         { fieldId = 1 }
                        else if sLen == 12 && matchBytes(bytes + strStart, "installments", 12) { fieldId = 2 }
                        else if sLen == 12 && matchBytes(bytes + strStart, "requested_at", 12) { fieldId = 3 }
                    } else if ctx == 2 {
                        if sLen == 10 && matchBytes(bytes + strStart, "avg_amount", 10)        { fieldId = 4 }
                        else if sLen == 12 && matchBytes(bytes + strStart, "tx_count_24h", 12) { fieldId = 5 }
                        else if sLen == 15 && matchBytes(bytes + strStart, "known_merchants", 15) { fieldId = 6 }
                    } else if ctx == 3 {
                        if sLen == 2 && matchBytes(bytes + strStart, "id", 2)          { fieldId = 7 }
                        else if sLen == 3 && matchBytes(bytes + strStart, "mcc", 3)    { fieldId = 8 }
                        else if sLen == 10 && matchBytes(bytes + strStart, "avg_amount", 10) { fieldId = 9 }
                    } else if ctx == 4 {
                        if sLen == 9 && matchBytes(bytes + strStart, "is_online", 9)     { fieldId = 15 }
                        else if sLen == 12 && matchBytes(bytes + strStart, "card_present", 12) { fieldId = 16 }
                        else if sLen == 12 && matchBytes(bytes + strStart, "km_from_home", 12) { fieldId = 17 }
                    } else if ctx == 5 {
                        if sLen == 9 && matchBytes(bytes + strStart, "timestamp", 9)        { fieldId = 18 }
                        else if sLen == 15 && matchBytes(bytes + strStart, "km_from_current", 15) { fieldId = 19 }
                    }
                } else {
                    // String value
                    if inKnownMerchants {
                        if knownCount < 32 {
                            let hash = fnv1a(bytes + strStart, strEnd - strStart)
                            withUnsafeMutablePointer(to: &knownHashes) { ptr in
                                let base = UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: UInt64.self)
                                base[knownCount] = hash
                            }
                            knownCount += 1
                        }
                    } else if ctx == 1 && fieldId == 3 {
                        parseTimestamp(bytes + strStart, strEnd - strStart, &txYear, &txMonth, &txDay, &txHour, &txMinute, &txSecond)
                    } else if ctx == 3 && fieldId == 7 {
                        merchantIdHash = fnv1a(bytes + strStart, strEnd - strStart)
                        hasMerchantId = true
                    } else if ctx == 3 && fieldId == 8 {
                        merchantMcc = parseIntFromBytes(bytes + strStart, strEnd - strStart)
                    } else if ctx == 5 && fieldId == 18 {
                        parseTimestamp(bytes + strStart, strEnd - strStart, &lastYear, &lastMonth, &lastDay, &lastHour, &lastMinute, &lastSecond)
                    }
                }

            case 0x6E: // 'n' for null
                if ctx == 0 && rootFieldId == 14 { hasLastTx = false }
                pos += 4

            case 0x74: // 't' for true
                if ctx == 4 {
                    if fieldId == 15 { isOnline = true }
                    else if fieldId == 16 { cardPresent = true }
                }
                pos += 4

            case 0x66: // 'f' for false
                if ctx == 4 {
                    if fieldId == 15 { isOnline = false }
                    else if fieldId == 16 { cardPresent = false }
                }
                pos += 5

            case 0x2C: // ,
                pos += 1

            case 0x3A: // :
                pos += 1

            default:
                // Number
                if (ch >= 0x30 && ch <= 0x39) || ch == 0x2D || ch == 0x2E {
                    let numStart = pos
                    pos += 1
                    while pos < len {
                        let nc = bytes[pos]
                        if (nc >= 0x30 && nc <= 0x39) || nc == 0x2E || nc == 0x2D || nc == 0x65 || nc == 0x45 || nc == 0x2B {
                            pos += 1
                        } else {
                            break
                        }
                    }
                    let numVal = parseDouble(bytes + numStart, pos - numStart)

                    switch ctx {
                    case 1:
                        if fieldId == 1 { txAmount = numVal }
                        else if fieldId == 2 { txInstallments = numVal }
                    case 2:
                        if fieldId == 4 { custAvgAmount = numVal }
                        else if fieldId == 5 { custTxCount24h = numVal }
                    case 3:
                        if fieldId == 9 { merchantAvgAmount = numVal }
                    case 4:
                        if fieldId == 17 { kmFromHome = numVal }
                    case 5:
                        if fieldId == 19 { lastKmFromCurrent = numVal }
                    default: break
                    }
                } else {
                    pos += 1
                }
            }
        }

        // Compute feature vector
        let dow = dayOfWeekMonBased(year: txYear, month: txMonth, day: txDay)

        var isUnknown = true
        if hasMerchantId && knownCount > 0 {
            withUnsafePointer(to: &knownHashes) { ptr in
                let base = UnsafeRawPointer(ptr).assumingMemoryBound(to: UInt64.self)
                for i in 0..<knownCount {
                    if base[i] == merchantIdHash { isUnknown = false; return }
                }
            }
        }

        if hasLastTx {
            let minutes: Double
            if txYear == lastYear && txMonth == lastMonth {
                let deltaSec = (txDay - lastDay) * 86400
                    + (txHour - lastHour) * 3600
                    + (txMinute - lastMinute) * 60
                    + (txSecond - lastSecond)
                minutes = Double(deltaSec) / 60.0
            } else {
                let txTotal = totalSeconds(txYear, txMonth, txDay, txHour, txMinute, txSecond)
                let lastTotal = totalSeconds(lastYear, lastMonth, lastDay, lastHour, lastMinute, lastSecond)
                minutes = Double(txTotal - lastTotal) / 60.0
            }
            vector[0] = round4(clamp01(lastKmFromCurrent / MccRisk.maxKm))
            vector[3] = round4(clamp01(minutes / MccRisk.maxMinutes))
        } else {
            vector[0] = -1.0
            vector[3] = -1.0
        }

        vector[1] = cardPresent ? 1.0 : 0.0
        vector[2] = isOnline ? 1.0 : 0.0
        vector[4] = isUnknown ? 1.0 : 0.0

        if custAvgAmount == 0 {
            vector[5] = 1.0
        } else {
            vector[5] = round4(clamp01((txAmount / custAvgAmount) / MccRisk.amountVsAvgRatio))
        }

        vector[6] = round4(Double(dow) / 6.0)
        vector[7] = round4(clamp01(kmFromHome / MccRisk.maxKm))
        vector[8] = round4(clamp01(txAmount / MccRisk.maxAmount))
        vector[9] = round4(clamp01(txInstallments / MccRisk.maxInstallments))
        vector[10] = round4(clamp01(custTxCount24h / MccRisk.maxTxCount24h))
        vector[11] = MccRisk.risk(for: merchantMcc)
        vector[12] = round4(Double(txHour) / 23.0)
        vector[13] = round4(clamp01(merchantAvgAmount / MccRisk.maxMerchantAvgAmount))
    }

    @inline(__always)
    private static func matchBytes(_ ptr: UnsafePointer<UInt8>, _ expected: StaticString, _ len: Int) -> Bool {
        expected.withUTF8Buffer { buf in
            memcmp(ptr, buf.baseAddress!, len) == 0
        }
    }

    @inline(__always)
    private static func parseTimestamp(_ ptr: UnsafePointer<UInt8>, _ len: Int,
                                       _ year: inout Int, _ month: inout Int, _ day: inout Int,
                                       _ hour: inout Int, _ minute: inout Int, _ second: inout Int) {
        guard len >= 19 else { return }
        year   = dig(ptr[0]) * 1000 + dig(ptr[1]) * 100 + dig(ptr[2]) * 10 + dig(ptr[3])
        month  = dig(ptr[5]) * 10 + dig(ptr[6])
        day    = dig(ptr[8]) * 10 + dig(ptr[9])
        hour   = dig(ptr[11]) * 10 + dig(ptr[12])
        minute = dig(ptr[14]) * 10 + dig(ptr[15])
        second = dig(ptr[17]) * 10 + dig(ptr[18])
    }

    @inline(__always)
    private static func dig(_ b: UInt8) -> Int { Int(b) - 48 }

    @inline(__always)
    private static func parseIntFromBytes(_ ptr: UnsafePointer<UInt8>, _ len: Int) -> Int {
        var result = 0
        for i in 0..<len {
            let d = Int(ptr[i]) - 48
            guard d >= 0 && d <= 9 else { return -1 }
            result = result * 10 + d
        }
        return result
    }

    @inline(__always)
    private static func parseDouble(_ ptr: UnsafePointer<UInt8>, _ len: Int) -> Double {
        var result: Double = 0
        var i = 0
        var negative = false

        if i < len && ptr[i] == 0x2D { negative = true; i += 1 }

        while i < len {
            let d = Int(ptr[i]) - 48
            guard d >= 0 && d <= 9 else { break }
            result = result * 10 + Double(d)
            i += 1
        }

        if i < len && ptr[i] == 0x2E {
            i += 1
            var frac: Double = 0
            var divisor: Double = 1
            while i < len {
                let d = Int(ptr[i]) - 48
                guard d >= 0 && d <= 9 else { break }
                frac = frac * 10 + Double(d)
                divisor *= 10
                i += 1
            }
            result += frac / divisor
        }

        if i < len && (ptr[i] == 0x65 || ptr[i] == 0x45) {
            i += 1
            var expNeg = false
            if i < len && ptr[i] == 0x2D { expNeg = true; i += 1 }
            else if i < len && ptr[i] == 0x2B { i += 1 }
            var exp = 0
            while i < len {
                let d = Int(ptr[i]) - 48
                guard d >= 0 && d <= 9 else { break }
                exp = exp * 10 + d
                i += 1
            }
            if expNeg {
                var div: Double = 1
                for _ in 0..<exp { div *= 10 }
                result /= div
            } else {
                var mul: Double = 1
                for _ in 0..<exp { mul *= 10 }
                result *= mul
            }
        }

        return negative ? -result : result
    }

    private static let cumulativeDays = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    @inline(__always)
    private static func totalSeconds(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int, _ second: Int) -> Int {
        let y = year - 1
        var days = y * 365 + y / 4 - y / 100 + y / 400
        days += cumulativeDays[month]
        if month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) {
            days += 1
        }
        days += day
        return days * 86400 + hour * 3600 + minute * 60 + second
    }
}
