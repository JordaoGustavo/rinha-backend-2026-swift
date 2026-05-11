import Foundation

public enum TransactionParser {
    private static let fnvOffsetBasis: UInt64 = 14695981039346656037
    private static let fnvPrime: UInt64 = 1099511628211

    @inline(__always)
    private static func fnv1a(_ string: String) -> UInt64 {
        var hash = fnvOffsetBasis
        for byte in string.utf8 {
            hash ^= UInt64(byte)
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

    /// Parse transaction JSON (nested format) and produce a 14-dim float vector (padded to 16).
    /// Dimensions are variance-reordered to match the reference implementation:
    /// [0]=kmCurrent [1]=cardPresent [2]=isOnline [3]=minsSinceLastTx
    /// [4]=unknownMerchant [5]=amtRatio [6]=dayOfWeek [7]=kmHome
    /// [8]=txAmount [9]=installments [10]=txCount24h [11]=mccRisk [12]=txHour [13]=merchAvg
    public static func parse(_ json: UnsafeRawBufferPointer, into vector: UnsafeMutablePointer<Float>) {
        for i in 0..<16 { vector[i] = 0 }
        guard json.count > 0 else { return }

        let data = Data(bytes: json.baseAddress!, count: json.count)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

        let transaction = obj["transaction"] as? [String: Any]
        let customer = obj["customer"] as? [String: Any]
        let merchant = obj["merchant"] as? [String: Any]
        let terminal = obj["terminal"] as? [String: Any]
        let lastTransaction = obj["last_transaction"] as? [String: Any]

        let txAmount = (transaction?["amount"] as? NSNumber)?.doubleValue ?? 0
        let txInstallments = (transaction?["installments"] as? NSNumber)?.doubleValue ?? 0
        let requestedAt = transaction?["requested_at"] as? String ?? ""

        let custAvgAmount = (customer?["avg_amount"] as? NSNumber)?.doubleValue ?? 0
        let custTxCount24h = (customer?["tx_count_24h"] as? NSNumber)?.doubleValue ?? 0
        let knownMerchants = customer?["known_merchants"] as? [String] ?? []

        let merchantId = merchant?["id"] as? String ?? ""
        let merchantMccRaw = merchant?["mcc"]
        let merchantAvgAmount = (merchant?["avg_amount"] as? NSNumber)?.doubleValue ?? 0

        let isOnline = (terminal?["is_online"] as? NSNumber)?.boolValue ?? false
        let cardPresent = (terminal?["card_present"] as? NSNumber)?.boolValue ?? false
        let kmFromHome = (terminal?["km_from_home"] as? NSNumber)?.doubleValue ?? 0

        var merchantMcc = -1
        if let mccStr = merchantMccRaw as? String, let v = Int(mccStr) {
            merchantMcc = v
        }

        var txYear = 0, txMonth = 0, txDay = 0, txHour = 0, txMinute = 0, txSecond = 0
        parseTimestamp(requestedAt, &txYear, &txMonth, &txDay, &txHour, &txMinute, &txSecond)

        let dow = dayOfWeekMonBased(year: txYear, month: txMonth, day: txDay)

        let merchantHash = fnv1a(merchantId)
        var isUnknown = true
        for km in knownMerchants {
            if fnv1a(km) == merchantHash {
                isUnknown = false
                break
            }
        }

        // [0] kmFromCurrent, [3] minsSinceLastTx
        if let lastTx = lastTransaction {
            let lastTs = lastTx["timestamp"] as? String ?? ""
            let kmFromCurrent = (lastTx["km_from_current"] as? NSNumber)?.doubleValue ?? 0

            var lastYear = 0, lastMonth = 0, lastDay = 0, lastHour = 0, lastMinute = 0, lastSecond = 0
            parseTimestamp(lastTs, &lastYear, &lastMonth, &lastDay, &lastHour, &lastMinute, &lastSecond)

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

            vector[0] = round4(clamp01(kmFromCurrent / MccRisk.maxKm))
            vector[3] = round4(clamp01(minutes / MccRisk.maxMinutes))
        } else {
            vector[0] = -1.0
            vector[3] = -1.0
        }

        // [1] cardPresent
        vector[1] = cardPresent ? 1.0 : 0.0

        // [2] isOnline
        vector[2] = isOnline ? 1.0 : 0.0

        // [4] unknownMerchant
        vector[4] = isUnknown ? 1.0 : 0.0

        // [5] amountRatio
        if custAvgAmount == 0 {
            vector[5] = 1.0
        } else {
            vector[5] = round4(clamp01((txAmount / custAvgAmount) / MccRisk.amountVsAvgRatio))
        }

        // [6] dayOfWeek / 6
        vector[6] = round4(Double(dow) / 6.0)

        // [7] kmFromHome / maxKm
        vector[7] = round4(clamp01(kmFromHome / MccRisk.maxKm))

        // [8] txAmount / maxAmount
        vector[8] = round4(clamp01(txAmount / MccRisk.maxAmount))

        // [9] installments / maxInstallments
        vector[9] = round4(clamp01(txInstallments / MccRisk.maxInstallments))

        // [10] txCount24h / maxTxCount24h
        vector[10] = round4(clamp01(custTxCount24h / MccRisk.maxTxCount24h))

        // [11] mccRisk
        vector[11] = MccRisk.risk(for: merchantMcc)

        // [12] txHour / 23
        vector[12] = round4(Double(txHour) / 23.0)

        // [13] merchantAvgAmount / maxMerchantAvgAmount
        vector[13] = round4(clamp01(merchantAvgAmount / MccRisk.maxMerchantAvgAmount))
    }

    @inline(__always)
    private static func parseTimestamp(_ ts: String, _ year: inout Int, _ month: inout Int, _ day: inout Int, _ hour: inout Int, _ minute: inout Int, _ second: inout Int) {
        let bytes = Array(ts.utf8)
        guard bytes.count >= 19 else { return }
        year = parseInt(bytes, 0, 4)
        month = parseInt(bytes, 5, 2)
        day = parseInt(bytes, 8, 2)
        hour = parseInt(bytes, 11, 2)
        minute = parseInt(bytes, 14, 2)
        second = parseInt(bytes, 17, 2)
    }

    @inline(__always)
    private static func parseInt(_ bytes: [UInt8], _ start: Int, _ length: Int) -> Int {
        var result = 0
        for i in start..<(start + length) {
            guard i < bytes.count else { break }
            let digit = Int(bytes[i]) - 48
            if digit >= 0 && digit <= 9 {
                result = result * 10 + digit
            }
        }
        return result
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
