import Testing
@testable import FraudDetector

@Test func goldenPayloadParsing() {
    // Initialize MccRisk with defaults (no file needed for this test)
    // Use default values that are already set

    let json = """
    {
      "transaction": {
        "amount": 250.00,
        "installments": 1,
        "requested_at": "2026-04-15T13:42:11Z"
      },
      "customer": {
        "avg_amount": 180.00,
        "tx_count_24h": 3,
        "known_merchants": ["m-001", "m-002"]
      },
      "merchant": {
        "id": "m-001",
        "mcc": "5411",
        "avg_amount": 200.00
      },
      "terminal": {
        "is_online": true,
        "card_present": true,
        "km_from_home": 2.5
      },
      "last_transaction": {
        "timestamp": "2026-04-15T11:10:00Z",
        "km_from_current": 1.8
      }
    }
    """

    var vector = [Float](repeating: 0, count: 16)
    let jsonBytes = Array(json.utf8)

    jsonBytes.withUnsafeBufferPointer { buf in
        vector.withUnsafeMutableBufferPointer { vecBuf in
            TransactionParser.parse(UnsafeRawBufferPointer(buf), into: vecBuf.baseAddress!)
        }
    }

    // [0] kmFromCurrent = round4(1.8/1000) = 0.0018
    #expect(vector[0] == Float(0.0018))
    // [1] cardPresent = 1.0
    #expect(vector[1] == Float(1.0))
    // [2] isOnline = 1.0
    #expect(vector[2] == Float(1.0))
    // [3] minsSinceLastTx: deltaSec = 2h32m11s = 9131s, mins = 152.1833
    //   round4(152.1833/1440) = round4(0.10568) = 0.1057
    #expect(vector[3] == Float(0.1057))
    // [4] unknownMerchant = 0.0 (m-001 is known)
    #expect(vector[4] == Float(0.0))
    // [5] amtRatio = round4((250/180)/10) = round4(0.13889) = 0.1389
    #expect(vector[5] == Float(0.1389))
    // [6] dayOfWeek: 2026-04-15 = Wednesday = monBased 2, round4(2/6) = 0.3333
    #expect(vector[6] == Float(0.3333))
    // [7] kmHome = round4(2.5/1000) = 0.0025
    #expect(vector[7] == Float(0.0025))
    // [8] txAmount = round4(250/10000) = 0.025
    #expect(vector[8] == Float(0.025))
    // [9] installments = round4(1/12) = 0.0833
    #expect(vector[9] == Float(0.0833))
    // [10] txCount24h = round4(3/20) = 0.15
    #expect(vector[10] == Float(0.15))
    // [11] mccRisk for 5411 (using default 0.50)
    #expect(vector[11] == Float(0.50))
    // [12] txHour = round4(13/23) = 0.5652
    #expect(vector[12] == Float(0.5652))
    // [13] merchAvg = round4(200/10000) = 0.02
    #expect(vector[13] == Float(0.02))
}

@Test func nullLastTransaction() {
    let json = """
    {
      "transaction": {"amount": 100, "installments": 0, "requested_at": "2026-01-01T12:00:00Z"},
      "customer": {"avg_amount": 100, "tx_count_24h": 1, "known_merchants": []},
      "merchant": {"id": "m-999", "mcc": "1234", "avg_amount": 50},
      "terminal": {"is_online": false, "card_present": false, "km_from_home": 0},
      "last_transaction": null
    }
    """

    var vector = [Float](repeating: 0, count: 16)
    let jsonBytes = Array(json.utf8)

    jsonBytes.withUnsafeBufferPointer { buf in
        vector.withUnsafeMutableBufferPointer { vecBuf in
            TransactionParser.parse(UnsafeRawBufferPointer(buf), into: vecBuf.baseAddress!)
        }
    }

    #expect(vector[0] == Float(-1.0))
    #expect(vector[3] == Float(-1.0))
    #expect(vector[1] == Float(0.0))
    #expect(vector[2] == Float(0.0))
    #expect(vector[4] == Float(1.0)) // unknown merchant
}
