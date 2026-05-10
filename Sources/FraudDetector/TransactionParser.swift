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
        if let mccNum = merchantMccRaw as? NSNumber {
            merchantMcc = mccNum.intValue
        } else if let mccStr = merchantMccRaw as? String, let v = Int(mccStr) {
            merchantMcc = v
        }

        var txYear = 0, txMonth = 0, txDay = 0, txHour = 0, txMinute = 0, txSecond = 0
        parseTimestamp(requestedAt, &txYear, &txMonth, &txDay, &txHour, &txMinute, &txSecond)

        // [0] amount / maxAmount
        vector[0] = round4(clamp01(txAmount / MccRisk.maxAmount))

        // [1] installments / maxInstallments
        vector[1] = round4(clamp01(txInstallments / MccRisk.maxInstallments))

        // [2] (amount / custAvgAmount) / amountVsAvgRatio
        if custAvgAmount == 0 {
            vector[2] = 1.0
        } else {
            vector[2] = round4(clamp01((txAmount / custAvgAmount) / MccRisk.amountVsAvgRatio))
        }

        // [3] hour / 23
        vector[3] = round4(Double(txHour) / 23.0)

        // [4] dayOfWeek / 6
        let dow = dayOfWeekMonBased(year: txYear, month: txMonth, day: txDay)
        vector[4] = round4(Double(dow) / 6.0)

        // [5] minutesSinceLastTx, [6] kmFromCurrent
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

            vector[5] = round4(clamp01(minutes / MccRisk.maxMinutes))
            vector[6] = round4(clamp01(kmFromCurrent / MccRisk.maxKm))
        } else {
            vector[5] = -1.0
            vector[6] = -1.0
        }

        // [7] kmFromHome / maxKm
        vector[7] = round4(clamp01(kmFromHome / MccRisk.maxKm))

        // [8] txCount24h / maxTxCount24h
        vector[8] = round4(clamp01(custTxCount24h / MccRisk.maxTxCount24h))

        // [9] isOnline
        vector[9] = isOnline ? 1.0 : 0.0

        // [10] cardPresent
        vector[10] = cardPresent ? 1.0 : 0.0

        // [11] isUnknownMerchant — check if merchant.id is in customer.known_merchants
        let merchantHash = fnv1a(merchantId)
        var isUnknown = true
        for km in knownMerchants {
            if fnv1a(km) == merchantHash {
                isUnknown = false
                break
            }
        }
        vector[11] = isUnknown ? 1.0 : 0.0

        // [12] mccRisk
        vector[12] = MccRisk.risk(for: merchantMcc)

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

    @inline(__always)
    private static func totalSeconds(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int, _ second: Int) -> Int {
        let totalDays = year * 365 + year / 4 - year / 100 + year / 400 + month * 30 + day
        return totalDays * 86400 + hour * 3600 + minute * 60 + second
    }
}
