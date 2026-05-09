import Foundation

enum GenWarmup {
    static func generate(outputPath: String, count: Int) throws {
        let amounts: [Double] = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000]
        let mccs: [Int] = [5411, 5812, 5814, 5912, 5921, 5942, 6011, 7011, 7995, 4121]
        let kmFromHome: [Double] = [0.2, 1.5, 5.0, 25.0, 200.0]

        let installmentsList: [Int] = [1, 1, 2, 3, 6, 12, 1, 1, 1, 2]
        let txCount24hList: [Int] = [0, 1, 3, 5, 10, 20, 2, 0, 7, 15]
        let terminalFlagsList: [Int] = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]

        let merchantIds = [
            "merchant_supermarket_01",
            "merchant_restaurant_02",
            "merchant_fastfood_03",
            "merchant_pharmacy_04",
            "merchant_liquor_05",
            "merchant_bookstore_06",
            "merchant_atm_07",
            "merchant_hotel_08",
            "merchant_gaming_09",
            "merchant_rideshare_10"
        ]

        let cardIds = [
            "card_visa_0001",
            "card_master_0002",
            "card_visa_0003",
            "card_amex_0004",
            "card_master_0005"
        ]

        // Build cross-product: amounts x mccs x kmFromHome = 10 * 10 * 5 = 500 combos
        // Cycle through the other fields
        var lines: [String] = []
        lines.reserveCapacity(count)

        var idx = 0
        outer: for amount in amounts {
            for mcc in mccs {
                for km in kmFromHome {
                    if idx >= count { break outer }

                    let installments = installmentsList[idx % installmentsList.count]
                    let txCount = txCount24hList[idx % txCount24hList.count]
                    let terminalFlag = terminalFlagsList[idx % terminalFlagsList.count]
                    let merchantId = merchantIds[idx % merchantIds.count]
                    let cardId = cardIds[idx % cardIds.count]

                    let payload: [String: Any] = [
                        "amount": amount,
                        "mcc": mcc,
                        "km_from_home": km,
                        "installments": installments,
                        "tx_count_24h": txCount,
                        "terminal_flag": terminalFlag,
                        "merchant_id": merchantId,
                        "card_id": cardId,
                        "hour_of_day": idx % 24,
                        "day_of_week": idx % 7,
                        "is_international": (idx % 5 == 0) ? true : false,
                        "card_age_days": 30 + (idx * 17) % 3600,
                        "days_since_last_tx": idx % 30,
                        "avg_amount_30d": amount * 0.8
                    ]

                    let jsonData = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
                    if let jsonString = String(data: jsonData, encoding: .utf8) {
                        lines.append(jsonString)
                    }

                    idx += 1
                }
            }
        }

        let output = lines.joined(separator: "\n") + "\n"
        try output.write(toFile: outputPath, atomically: true, encoding: .utf8)

        print("Generated \(lines.count) warmup payloads to \(outputPath)")
    }
}
