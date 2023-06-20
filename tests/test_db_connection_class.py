import json
import pytest

from unittest.mock import Mock
from unittest.mock import patch

from fastapi_service.db_connection import DBConnection
from fastapi_service.transaction import Transaction

trn_id = 3446051
transaction_json = '{"TransactionID": 3446051, "isFraud": 1, "TransactionDT": 11757454, "TransactionAmt": 311.95, ' \
                  '"ProductCD": "W", "card1": 6344, "card2": 327.0, "card3": 150.0, "card4": "discover", "card5": 224.0, ' \
                  '"card6": "credit", "addr1": 264.0, "addr2": 87.0, "dist1": NaN, "dist2": NaN, ' \
                  '"P_emaildomain": "yahoo.com", "R_emaildomain": NaN, "C1": 205.0, "C2": 190.0, "C3": 0.0, "C4": 0.0, ' \
                  '"C5": 154.0, "C6": 186.0, "C7": 0.0, "C8": 0.0, "C9": 80.0, "C10": 0.0, "C11": 183.0, "C12": 0.0, ' \
                  '"C13": 240.0, "C14": 104.0, "D1": 3.0, "D2": 3.0, "D3": 1.0, "D4": 2.0, "D5": 2.0, "D6": NaN, ' \
                  '"D7": NaN, "D8": NaN, "D9": NaN, "D10": 2.0, "D11": 0.0, "D12": NaN, "D13": NaN, "D14": NaN, ' \
                  '"D15": 2.0, "M1": "T", "M2": "F", "M3": "F", "M4": NaN, "M5": NaN, "M6": "F", "M7": "F", "M8": "F", ' \
                  '"M9": "F", "V1": 1.0, "V2": 1.0, "V3": 1.0, "V4": 1.0, "V5": 1.0, "V6": 1.0, "V7": 1.0, "V8": 1.0, ' \
                  '"V9": 1.0, "V10": 1.0, "V11": 1.0, "V12": 1.0, "V13": 2.0, "V14": 1.0, "V15": 0.0, "V16": 0.0, ' \
                  '"V17": 0.0, "V18": 0.0, "V19": 1.0, "V20": 1.0, "V21": 0.0, "V22": 0.0, "V23": 1.0, "V24": 1.0, ' \
                  '"V25": 1.0, "V26": 1.0, "V27": 0.0, "V28": 0.0, "V29": 1.0, "V30": 2.0, "V31": 0.0, "V32": 0.0, ' \
                  '"V33": 0.0, "V34": 0.0, "V35": 1.0, "V36": 2.0, "V37": 1.0, "V38": 1.0, "V39": 0.0, "V40": 0.0, ' \
                  '"V41": 1.0, "V42": 0.0, "V43": 0.0, "V44": 1.0, "V45": 1.0, "V46": 1.0, "V47": 1.0, "V48": 1.0, ' \
                  '"V49": 2.0, "V50": 0.0, "V51": 0.0, "V52": 0.0, "V53": 1.0, "V54": 2.0, "V55": 1.0, "V56": 1.0, ' \
                  '"V57": 0.0, "V58": 0.0, "V59": 0.0, "V60": 0.0, "V61": 1.0, "V62": 1.0, "V63": 0.0, "V64": 0.0, ' \
                  '"V65": 1.0, "V66": 1.0, "V67": 1.0, "V68": 0.0, "V69": 1.0, "V70": 2.0, "V71": 0.0, "V72": 0.0, ' \
                  '"V73": 0.0, "V74": 0.0, "V75": 1.0, "V76": 2.0, "V77": 1.0, "V78": 1.0, "V79": 0.0, "V80": 0.0, ' \
                  '"V81": 0.0, "V82": 1.0, "V83": 1.0, "V84": 0.0, "V85": 0.0, "V86": 1.0, "V87": 1.0, "V88": 1.0, ' \
                  '"V89": 0.0, "V90": 1.0, "V91": 2.0, "V92": 0.0, "V93": 0.0, "V94": 0.0, "V95": 0.0, "V96": 3.0, ' \
                  '"V97": 3.0, "V98": 0.0, "V99": 1.0, "V100": 1.0, "V101": 0.0, "V102": 2.0, "V103": 2.0, "V104": 0.0, ' \
                  '"V105": 0.0, "V106": 0.0, "V107": 1.0, "V108": 1.0, "V109": 1.0, "V110": 1.0, "V111": 1.0, ' \
                  '"V112": 1.0, "V113": 1.0, "V114": 1.0, "V115": 1.0, "V116": 1.0, "V117": 1.0, "V118": 1.0, ' \
                  '"V119": 1.0, "V120": 1.0, "V121": 1.0, "V122": 1.0, "V123": 1.0, "V124": 1.0, "V125": 1.0, ' \
                  '"V126": 0.0, "V127": 935.8499755859376, "V128": 935.8499755859376, "V129": 0.0, ' \
                  '"V130": 311.95001220703125, "V131": 311.95001220703125, "V132": 0.0, "V133": 623.9000244140625, ' \
                  '"V134": 623.9000244140625, "V135": 0.0, "V136": 0.0, "V137": 0.0, "V138": NaN, "V139": NaN, ' \
                  '"V140": NaN, "V141": NaN, "V142": NaN, "V143": NaN, "V144": NaN, "V145": NaN, "V146": NaN, ' \
                  '"V147": NaN, "V148": NaN, "V149": NaN, "V150": NaN, "V151": NaN, "V152": NaN, "V153": NaN, ' \
                  '"V154": NaN, "V155": NaN, "V156": NaN, "V157": NaN, "V158": NaN, "V159": NaN, "V160": NaN, ' \
                  '"V161": NaN, "V162": NaN, "V163": NaN, "V164": NaN, "V165": NaN, "V166": NaN, "V167": NaN, ' \
                  '"V168": NaN, "V169": NaN, "V170": NaN, "V171": NaN, "V172": NaN, "V173": NaN, "V174": NaN, ' \
                  '"V175": NaN, "V176": NaN, "V177": NaN, "V178": NaN, "V179": NaN, "V180": NaN, "V181": NaN, ' \
                  '"V182": NaN, "V183": NaN, "V184": NaN, "V185": NaN, "V186": NaN, "V187": NaN, "V188": NaN, ' \
                  '"V189": NaN, "V190": NaN, "V191": NaN, "V192": NaN, "V193": NaN, "V194": NaN, "V195": NaN, ' \
                  '"V196": NaN, "V197": NaN, "V198": NaN, "V199": NaN, "V200": NaN, "V201": NaN, "V202": NaN, ' \
                  '"V203": NaN, "V204": NaN, "V205": NaN, "V206": NaN, "V207": NaN, "V208": NaN, "V209": NaN, ' \
                  '"V210": NaN, "V211": NaN, "V212": NaN, "V213": NaN, "V214": NaN, "V215": NaN, "V216": NaN, ' \
                  '"V217": NaN, "V218": NaN, "V219": NaN, "V220": NaN, "V221": NaN, "V222": NaN, "V223": NaN, ' \
                  '"V224": NaN, "V225": NaN, "V226": NaN, "V227": NaN, "V228": NaN, "V229": NaN, "V230": NaN, ' \
                  '"V231": NaN, "V232": NaN, "V233": NaN, "V234": NaN, "V235": NaN, "V236": NaN, "V237": NaN, ' \
                  '"V238": NaN, "V239": NaN, "V240": NaN, "V241": NaN, "V242": NaN, "V243": NaN, "V244": NaN, ' \
                  '"V245": NaN, "V246": NaN, "V247": NaN, "V248": NaN, "V249": NaN, "V250": NaN, "V251": NaN, ' \
                  '"V252": NaN, "V253": NaN, "V254": NaN, "V255": NaN, "V256": NaN, "V257": NaN, "V258": NaN, ' \
                  '"V259": NaN, "V260": NaN, "V261": NaN, "V262": NaN, "V263": NaN, "V264": NaN, "V265": NaN, ' \
                  '"V266": NaN, "V267": NaN, "V268": NaN, "V269": NaN, "V270": NaN, "V271": NaN, "V272": NaN, ' \
                  '"V273": NaN, "V274": NaN, "V275": NaN, "V276": NaN, "V277": NaN, "V278": NaN, "V279": 0.0, ' \
                  '"V280": 3.0, "V281": 1.0, "V282": 2.0, "V283": 4.0, "V284": 0.0, "V285": 1.0, "V286": 0.0, ' \
                  '"V287": 1.0, "V288": 1.0, "V289": 1.0, "V290": 1.0, "V291": 1.0, "V292": 1.0, "V293": 0.0, ' \
                  '"V294": 2.0, "V295": 2.0, "V296": 0.0, "V297": 0.0, "V298": 0.0, "V299": 0.0, "V300": 0.0, ' \
                  '"V301": 0.0, "V302": 0.0, "V303": 0.0, "V304": 0.0, "V305": 1.0, "V306": 0.0, ' \
                  '"V307": 935.8499755859376, "V308": 935.8499755859376, "V309": 0.0, "V310": 311.95001220703125, ' \
                  '"V311": 0.0, "V312": 311.95001220703125, "V313": 311.95001220703125, "V314": 311.95001220703125, ' \
                  '"V315": 311.95001220703125, "V316": 0.0, "V317": 623.9000244140625, "V318": 623.9000244140625, ' \
                  '"V319": 0.0, "V320": 0.0, "V321": 0.0, "V322": NaN, "V323": NaN, "V324": NaN, "V325": NaN, ' \
                  '"V326": NaN, "V327": NaN, "V328": NaN, "V329": NaN, "V330": NaN, "V331": NaN, "V332": NaN, ' \
                  '"V333": NaN, "V334": NaN, "V335": NaN, "V336": NaN, "V337": NaN, "V338": NaN, "V339": NaN, ' \
                  '"id_01": NaN, "id_02": NaN, "id_03": NaN, "id_04": NaN, "id_05": NaN, "id_06": NaN, "id_07": NaN, ' \
                  '"id_08": NaN, "id_09": NaN, "id_10": NaN, "id_11": NaN, "id_12": NaN, "id_13": NaN, "id_14": NaN, ' \
                  '"id_15": NaN, "id_16": NaN, "id_17": NaN, "id_18": NaN, "id_19": NaN, "id_20": NaN, "id_21": NaN, ' \
                  '"id_22": NaN, "id_23": NaN, "id_24": NaN, "id_25": NaN, "id_26": NaN, "id_27": NaN, "id_28": NaN, ' \
                  '"id_29": NaN, "id_30": NaN, "id_31": NaN, "id_32": NaN, "id_33": NaN, "id_34": NaN, "id_35": NaN, ' \
                  '"id_36": NaN, "id_37": NaN, "id_38": NaN, "DeviceType": NaN, "DeviceInfo": NaN}'


async def get_transaction():
    return transaction_json


@pytest.mark.asyncio
@patch('aioredis.Redis.from_url')
async def test_get(mock_redis_from_url):
    redis_mock = Mock()
    redis_mock.get = Mock(return_value=get_transaction())
    mock_redis_from_url.return_value = redis_mock

    db_connection = DBConnection()
    actual = await db_connection.get(trn_id)
    expected = Transaction.parse_obj(json.loads(transaction_json))

    redis_mock.get.assert_called_once_with(trn_id)
    assert expected.TransactionID == actual.TransactionID
    assert expected == actual


async def get_keys():
    return [trn_id]

@pytest.mark.asyncio
@patch('aioredis.Redis.from_url')
async def test_get_all(mock_redis_from_url):
    redis_mock = Mock()
    redis_mock.get = Mock(return_value=get_transaction())
    redis_mock.keys = Mock(return_value=get_keys())
    mock_redis_from_url.return_value = redis_mock

    db_connection = DBConnection()
    actual = await db_connection.get_all()
    expected = [Transaction.parse_obj(json.loads(transaction_json))]

    redis_mock.keys.assert_called_once()
    redis_mock.get.assert_called_once_with(trn_id)
    assert len(expected) == len(actual)
    assert expected[0] == actual[0]


async def coroutine_for_redis_set():
    return


@pytest.mark.asyncio
@patch('aioredis.Redis.from_url')
async def test_put(mock_redis_from_url):
    redis_mock = Mock()
    redis_mock.set = Mock(return_value=coroutine_for_redis_set())
    mock_redis_from_url.return_value = redis_mock

    transaction = Transaction.parse_obj(json.loads(transaction_json))
    db_connection = DBConnection()
    await db_connection.put(transaction)

    redis_mock.set.assert_called_once_with(transaction.TransactionID, transaction_json)
