import Foundation

public enum TransactionParser {
    // FNV-1a constants
    private static let fnvOffsetBasis: UInt64 = 14695981039346656037
    private static let fnvPrime: UInt64 = 1099511628211

    // Known merchant IDs set (populated at startup)
    nonisolated(unsafe) private static var knownMerchantHashes = Set<UInt64>()

    /// Load known merchant IDs from a newline-delimited file.
    public static func loadKnownMerchants(from path: String) throws {
        let data = try String(contentsOfFile: path, encoding: .utf8)
        var hashes = Set<UInt64>()
        for line in data.split(separator: "\n", omittingEmptySubsequences: true) {
            hashes.insert(fnv1a(String(line)))
        }
        knownMerchantHashes = hashes
    }

    /// Load known merchant IDs from an array of strings.
    public static func loadKnownMerchants(_ merchants: [String]) {
        var hashes = Set<UInt64>()
        for m in merchants {
            hashes.insert(fnv1a(m))
        }
        knownMerchantHashes = hashes
    }

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
    private static func fnv1a(_ bytes: UnsafeRawBufferPointer, from start: Int, count: Int) -> UInt64 {
        var hash = fnvOffsetBasis
        for i in start..<(start + count) {
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

    /// Zeller's congruence returning Monday-based day (Mon=0..Sun=6).
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
        // Zeller's formula: h = (day + (13*(m+1))/5 + k + k/4 + j/4 - 2*j) mod 7
        // h: 0=Sat, 1=Sun, 2=Mon, 3=Tue, 4=Wed, 5=Thu, 6=Fri
        var h = (day + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 - 2 * j) % 7
        if h < 0 { h += 7 }
        // Convert to Monday-based: Mon=0, Tue=1, ..., Sun=6
        return (h + 5) % 7
    }

    /// Parse transaction JSON and produce a 14-dim float vector (padded to 16).
    /// The vector pointer must have room for at least 16 floats.
    public static func parse(_ json: UnsafeRawBufferPointer, into vector: UnsafeMutablePointer<Float>) {
        // Zero out all 16 floats (14 dims + 2 padding)
        for i in 0..<16 { vector[i] = 0 }

        guard json.count > 0 else { return }

        // Parse using JSONSerialization
        let data = Data(bytes: json.baseAddress!, count: json.count)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

        // Extract fields
        let amount = (obj["amount"] as? NSNumber)?.doubleValue ?? 0
        let installments = (obj["installments"] as? NSNumber)?.doubleValue ?? 1
        let mcc = (obj["mcc"] as? NSNumber)?.intValue ?? 0
        let merchantId = obj["merchant_id"] as? String ?? ""
        let merchantAvgAmount = (obj["merchant_avg_amount"] as? NSNumber)?.doubleValue ?? 0
        let custAvgAmount = (obj["customer_avg_amount"] as? NSNumber)?.doubleValue ?? 0
        let isOnline = (obj["is_online"] as? NSNumber)?.boolValue ?? false
        let cardPresent = (obj["card_present"] as? NSNumber)?.boolValue ?? false
        let txCount24h = (obj["tx_count_24h"] as? NSNumber)?.doubleValue ?? 0
        let kmFromHome = (obj["km_from_home"] as? NSNumber)?.doubleValue ?? 0

        // Timestamp parsing
        let timestamp = obj["timestamp"] as? String ?? ""
        var hour = 0
        var year = 2024
        var month = 1
        var day = 1
        parseTimestamp(timestamp, &year, &month, &day, &hour)

        // Last transaction fields
        let lastTxTimestamp = obj["last_tx_timestamp"] as? String
        let kmFromLastTx = obj["km_from_last_tx"] as? NSNumber

        // [0] amount / maxAmount (clamped 0-1)
        vector[0] = round4(clamp01(amount / MccRisk.maxAmount))

        // [1] installments / maxInstallments (clamped 0-1)
        vector[1] = round4(clamp01(installments / MccRisk.maxInstallments))

        // [2] (amount / custAvgAmount) / amountVsAvgRatio (clamped 0-1, or 1.0 if custAvgAmount==0)
        if custAvgAmount == 0 {
            vector[2] = 1.0
        } else {
            vector[2] = round4(clamp01((amount / custAvgAmount) / MccRisk.amountVsAvgRatio))
        }

        // [3] hour / 23
        vector[3] = round4(Double(hour) / 23.0)

        // [4] dayOfWeek(monBased) / 6
        let dow = dayOfWeekMonBased(year: year, month: month, day: day)
        vector[4] = round4(Double(dow) / 6.0)

        // [5] minutesSinceLastTx / maxMinutes (clamped 0-1, or -1 if no lastTx)
        if let lastTxTs = lastTxTimestamp, !lastTxTs.isEmpty {
            let minutes = minutesBetween(lastTxTs, timestamp)
            vector[5] = round4(clamp01(minutes / MccRisk.maxMinutes))
        } else {
            vector[5] = -1.0
        }

        // [6] kmFromCurrent / maxKm (clamped 0-1, or -1 if no lastTx)
        if let km = kmFromLastTx {
            vector[6] = round4(clamp01(km.doubleValue / MccRisk.maxKm))
        } else {
            vector[6] = -1.0
        }

        // [7] kmFromHome / maxKm (clamped 0-1)
        vector[7] = round4(clamp01(kmFromHome / MccRisk.maxKm))

        // [8] txCount24h / maxTxCount24h (clamped 0-1)
        vector[8] = round4(clamp01(txCount24h / MccRisk.maxTxCount24h))

        // [9] isOnline ? 1 : 0
        vector[9] = isOnline ? 1.0 : 0.0

        // [10] cardPresent ? 1 : 0
        vector[10] = cardPresent ? 1.0 : 0.0

        // [11] isUnknownMerchant ? 1 : 0
        let merchantHash = fnv1a(merchantId)
        vector[11] = knownMerchantHashes.contains(merchantHash) ? 0.0 : 1.0

        // [12] mccRisk[mcc]
        vector[12] = MccRisk.risk(for: mcc)

        // [13] merchantAvgAmount / maxMerchantAvgAmount (clamped 0-1)
        vector[13] = round4(clamp01(merchantAvgAmount / MccRisk.maxMerchantAvgAmount))

        // [14-15] padding (already zeroed)
    }

    /// Parse an ISO-8601-like timestamp "YYYY-MM-DDTHH:MM:SS..." to extract year, month, day, hour.
    @inline(__always)
    private static func parseTimestamp(_ ts: String, _ year: inout Int, _ month: inout Int, _ day: inout Int, _ hour: inout Int) {
        // Expected format: "2024-01-15T14:30:00" or similar
        // Positions:        0123456789012345678
        let bytes = Array(ts.utf8)
        guard bytes.count >= 13 else { return }

        year = parseInt(bytes, 0, 4)
        month = parseInt(bytes, 5, 2)
        day = parseInt(bytes, 8, 2)
        hour = parseInt(bytes, 11, 2)
    }

    @inline(__always)
    private static func parseInt(_ bytes: [UInt8], _ start: Int, _ length: Int) -> Int {
        var result = 0
        for i in start..<(start + length) {
            guard i < bytes.count else { break }
            let digit = Int(bytes[i]) - 48 // '0' = 48
            if digit >= 0 && digit <= 9 {
                result = result * 10 + digit
            }
        }
        return result
    }

    /// Calculate minutes between two ISO timestamps (same month/year scalar approach).
    /// Uses a simplified calculation: converts each to total minutes since epoch-ish,
    /// then takes the absolute difference.
    private static func minutesBetween(_ ts1: String, _ ts2: String) -> Double {
        let m1 = timestampToMinutes(ts1)
        let m2 = timestampToMinutes(ts2)
        return abs(m2 - m1)
    }

    @inline(__always)
    private static func timestampToMinutes(_ ts: String) -> Double {
        let bytes = Array(ts.utf8)
        guard bytes.count >= 16 else { return 0 }

        let year = parseInt(bytes, 0, 4)
        let month = parseInt(bytes, 5, 2)
        let day = parseInt(bytes, 8, 2)
        let hour = parseInt(bytes, 11, 2)
        let minute = parseInt(bytes, 14, 2)

        // Simple scalar: total minutes
        let totalDays = year * 365 + month * 30 + day
        return Double(totalDays * 1440 + hour * 60 + minute)
    }
}
