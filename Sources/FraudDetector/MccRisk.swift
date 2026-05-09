import Foundation

public enum MccRisk {
    private static let tableSize = 10000
    private static let defaultRisk: Float = 0.50
    // nonisolated(unsafe) because this is write-once at startup, read-only after
    nonisolated(unsafe) private static var _table = [Float](repeating: 0.50, count: tableSize)

    nonisolated(unsafe) public static var maxAmount: Double = 10000.0
    nonisolated(unsafe) public static var maxInstallments: Double = 12.0
    nonisolated(unsafe) public static var amountVsAvgRatio: Double = 10.0
    nonisolated(unsafe) public static var maxMinutes: Double = 1440.0
    nonisolated(unsafe) public static var maxKm: Double = 1000.0
    nonisolated(unsafe) public static var maxTxCount24h: Double = 20.0
    nonisolated(unsafe) public static var maxMerchantAvgAmount: Double = 10000.0

    @inline(__always)
    public static func risk(for mcc: Int) -> Float {
        guard mcc >= 0, mcc < tableSize else { return defaultRisk }
        return _table[mcc]
    }

    public static func initialize(mccRiskPath: String, normalizationPath: String) throws {
        _table = [Float](repeating: defaultRisk, count: tableSize)
        let mccData = try Data(contentsOf: URL(fileURLWithPath: mccRiskPath))
        if let mccJson = try JSONSerialization.jsonObject(with: mccData) as? [String: Any] {
            for (key, value) in mccJson {
                if let mcc = Int(key), mcc >= 0, mcc < tableSize, let risk = (value as? NSNumber)?.floatValue {
                    _table[mcc] = risk
                }
            }
        }
        let normData = try Data(contentsOf: URL(fileURLWithPath: normalizationPath))
        if let normJson = try JSONSerialization.jsonObject(with: normData) as? [String: Any] {
            maxAmount = (normJson["max_amount"] as? NSNumber)?.doubleValue ?? 10000.0
            maxInstallments = (normJson["max_installments"] as? NSNumber)?.doubleValue ?? 12.0
            amountVsAvgRatio = (normJson["amount_vs_avg_ratio"] as? NSNumber)?.doubleValue ?? 10.0
            maxMinutes = (normJson["max_minutes"] as? NSNumber)?.doubleValue ?? 1440.0
            maxKm = (normJson["max_km"] as? NSNumber)?.doubleValue ?? 1000.0
            maxTxCount24h = (normJson["max_tx_count_24h"] as? NSNumber)?.doubleValue ?? 20.0
            maxMerchantAvgAmount = (normJson["max_merchant_avg_amount"] as? NSNumber)?.doubleValue ?? 10000.0
        }
    }
}
