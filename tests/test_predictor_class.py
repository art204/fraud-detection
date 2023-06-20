import json
from collections import defaultdict

import numpy as np
import pandas as pd

from unittest.mock import Mock
from unittest.mock import patch

from fastapi_service.predictor import Predictor
from fastapi_service.transaction import Transaction


initial_trn_json = '{"TransactionID": 3446051, "isFraud": 1, "TransactionDT": 11757454, "TransactionAmt": 311.95, ' \
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

prepared_trn_json = '{"TransactionDT": 0.7422100893172672, "TransactionAmt": 0.009759765589529934, ' \
                    '"card1": 0.12681159420289856, "card2": 0.08872366790582403, "card3": 0.02455903836420205, ' \
                    '"card5": 0.038621300510063836, "addr1": 0.018549271502428324, "addr2": 0.023941953077135616, ' \
                    '"dist1": 0.0, "C1": 0.04375667022411953, "C3": 0.0, "C5": 0.44126074498567336, ' \
                    '"C13": 0.08224811514736122, "D1": 0.6113483831604637, "D3": 0.55005500550055, ' \
                    '"D4": 0.5373054213633924, "D5": 0.5506050605060505, "D10": 0.5338666666666667, ' \
                    '"D11": 0.5985620131815459, "D15": 0.5330138445154421, "V1": 1.0, "V2": 0.9930486593843099, ' \
                    '"V3": 0.992063492063492, "V4": 0.9950248756218905, "V6": 0.992063492063492, ' \
                    '"V7": 0.992063492063492, "V8": 0.9930486593843099, "V9": 0.9930486593843099, ' \
                    '"V10": 0.9970089730807576, "V12": 0.998003992015968, "V14": 1.0, "V15": 0.993041749502982, ' \
                    '"V17": 0.985207100591716, "V19": 0.9940357852882702, "V23": 0.9881422924901185, ' \
                    '"V24": 0.9881422924901185, "V25": 0.9940357852882702, "V26": 0.9881422924901185, ' \
                    '"V27": 0.9960119641076769, "V29": 0.9960159362549801, "V32": 0.985207100591716, ' \
                    '"V35": 0.998003992015968, "V37": 0.9496676163342831, "V38": 0.9496676163342831, ' \
                    '"V39": 0.985207100591716, "V41": 1.0, "V43": 0.9920556107249255, "V44": 0.9551098376313276, ' \
                    '"V46": 0.9950248756218905, "V47": 0.9891196834817012, "V49": 0.9970119521912351, ' \
                    '"V52": 0.9881305637982195, "V53": 0.9960159362549801, "V55": 0.984251968503937, ' \
                    '"V56": 0.9523809523809523, "V59": 0.9842364532019704, "V61": 0.9950248756218905, ' \
                    '"V62": 0.9910802775024777, "V66": 0.9950248756218905, "V67": 0.9930486593843099, ' \
                    '"V68": 0.998001998001998, "V70": 0.9960199004975123, "V74": 0.9920556107249255, ' \
                    '"V75": 0.9970089730807576, "V77": 0.9718172983479105, "V78": 0.970873786407767, ' \
                    '"V81": 0.9813359528487229, "V82": 0.9940357852882702, "V83": 0.9940357852882702, ' \
                    '"V86": 0.9718172983479105, "V87": 0.9718172983479105, "V95": 0.5319488817891375, ' \
                    '"V98": 0.9881305637982195, "V99": 0.9199632014719411, "V100": 0.9737098344693281, ' \
                    '"V104": 0.985207100591716, "V105": 0.9098360655737705, "V107": 1.0, "V108": 0.9940357852882702, ' \
                    '"V109": 0.9940357852882702, "V110": 0.9940357852882702, "V111": 0.992063492063492, ' \
                    '"V112": 0.992063492063492, "V114": 0.9950248756218905, "V115": 0.9950248756218905, ' \
                    '"V116": 0.9950248756218905, "V117": 0.9990009990009989, "V118": 0.9990009990009989, ' \
                    '"V119": 0.9990009990009989, "V120": 0.998003992015968, "V121": 0.998003992015968, ' \
                    '"V122": 0.998003992015968, "V123": 0.9881422924901185, "V124": 0.9881422924901185, ' \
                    '"V125": 0.9881422924901185, "V129": 0.017799871712636306, "V130": 0.02335810013910326, ' \
                    '"V131": 0.02335810013910326, "V135": 0.010888402053428375, "V136": 0.010888402053428375, ' \
                    '"V169": 0.0, "V170": 0.0, "V171": 0.0, "V172": 0.0, "V173": 0.0, "V174": 0.0, "V175": 0.0, ' \
                    '"V176": 0.0, "V180": 0.0, "V181": 0.0, "V184": 0.0, "V185": 0.0, "V186": 0.0, "V187": 0.0, ' \
                    '"V188": 0.0, "V189": 0.0, "V191": 0.0, "V194": 0.0, "V195": 0.0, "V199": 0.0, "V200": 0.0, ' \
                    '"V204": 0.0, "V205": 0.0, "V207": 0.0, "V208": 0.0, "V209": 0.0, "V210": 0.0, "V214": 0.0, ' \
                    '"V215": 0.0, "V216": 0.0, "V217": 0.0, "V220": 0.0, "V221": 0.0, "V223": 0.0, "V224": 0.0, ' \
                    '"V226": 0.0, "V227": 0.0, "V228": 0.0, "V229": 0.0, "V230": 0.0, "V234": 0.0, "V238": 0.0, ' \
                    '"V240": 0.0, "V241": 0.0, "V242": 0.0, "V245": 0.0, "V246": 0.0, "V247": 0.0, "V248": 0.0, "' \
                    'V250": 0.0, "V252": 0.0, "V255": 0.0, "V258": 0.0, "V260": 0.0, "V261": 0.0, "V262": 0.0, ' \
                    '"V264": 0.0, "V267": 0.0, "V268": 0.0, "V270": 0.0, "V274": 0.0, "V281": 0.9794319294809011, ' \
                    '"V282": 0.9709020368574199, "V283": 0.9400187441424555, "V284": 0.9881305637982195, ' \
                    '"V285": 0.9140767824497258, "V286": 0.9920556107249255, "V287": 0.970873786407767, ' \
                    '"V288": 0.9910802775024777, "V289": 0.9891196834817012, "V290": 0.9380863039399625, ' \
                    '"V291": 0.48685491723466406, "V300": 0.9891089108910891, "V303": 0.9803729146221786, ' \
                    '"V305": 0.9990009990009989, "V310": 0.02335810013910326, "V311": 0.017799871712636306, ' \
                    '"V312": 0.02335810013910326, "V313": 0.22538583776488016, "V314": 0.1538877802071527, ' \
                    '"V319": 0.009508942594161376, "V320": 0.009508942594161376, "id_01": 0.0, "id_02": 0.0, ' \
                    '"id_05": 0.0, "id_06": 0.0, "id_11": 0.0, "id_13": 0.021768773609614767, ' \
                    '"id_17": 0.021387512443467146, "id_19": 0.021404430051597985, "id_20": 0.0, ' \
                    '"id_31": 0.02116657914405666, "DeviceInfo": 0.025589502286433825, "ProductCD_H": 0.0, ' \
                    '"ProductCD_R": 0.0, "ProductCD_S": 0.0, "ProductCD_W": 1.0, "card4_american express": 0.0, ' \
                    '"card4_discover": 1.0, "card4_mastercard": 0.0, "card4_visa": 0.0, "card6_charge card": 0.0, ' \
                    '"card6_credit": 1.0, "card6_debit": 0.0, "card6_debit or credit": 0.0, ' \
                    '"P_emaildomain_aim.com": 0.0, "P_emaildomain_anonymous.com": 0.0, ' \
                    '"P_emaildomain_aol.com": 0.0, "P_emaildomain_att.net": 0.0, "P_emaildomain_bellsouth.net": 0.0, ' \
                    '"P_emaildomain_cableone.net": 0.0, "P_emaildomain_centurylink.net": 0.0, ' \
                    '"P_emaildomain_cfl.rr.com": 0.0, "P_emaildomain_charter.net": 0.0, ' \
                    '"P_emaildomain_comcast.net": 0.0, "P_emaildomain_cox.net": 0.0, ' \
                    '"P_emaildomain_earthlink.net": 0.0, "P_emaildomain_embarqmail.com": 0.0, ' \
                    '"P_emaildomain_frontier.com": 0.0, "P_emaildomain_frontiernet.net": 0.0, ' \
                    '"P_emaildomain_gmail": 0.0, "P_emaildomain_gmail.com": 0.0, "P_emaildomain_gmx.de": 0.0, ' \
                    '"P_emaildomain_hotmail.co.uk": 0.0, "P_emaildomain_hotmail.com": 0.0, ' \
                    '"P_emaildomain_hotmail.de": 0.0, "P_emaildomain_hotmail.es": 0.0, ' \
                    '"P_emaildomain_hotmail.fr": 0.0, "P_emaildomain_icloud.com": 0.0, "P_emaildomain_juno.com": 0.0, ' \
                    '"P_emaildomain_live.com": 0.0, "P_emaildomain_live.com.mx": 0.0, "P_emaildomain_live.fr": 0.0, ' \
                    '"P_emaildomain_mac.com": 0.0, "P_emaildomain_mail.com": 0.0, "P_emaildomain_me.com": 0.0, ' \
                    '"P_emaildomain_msn.com": 0.0, "P_emaildomain_netzero.com": 0.0, "P_emaildomain_netzero.net": 0.0, ' \
                    '"P_emaildomain_optonline.net": 0.0, "P_emaildomain_outlook.com": 0.0, ' \
                    '"P_emaildomain_outlook.es": 0.0, "P_emaildomain_prodigy.net.mx": 0.0, ' \
                    '"P_emaildomain_protonmail.com": 0.0, "P_emaildomain_ptd.net": 0.0, ' \
                    '"P_emaildomain_q.com": 0.0, "P_emaildomain_roadrunner.com": 0.0, ' \
                    '"P_emaildomain_rocketmail.com": 0.0, "P_emaildomain_sbcglobal.net": 0.0, ' \
                    '"P_emaildomain_sc.rr.com": 0.0, "P_emaildomain_servicios-ta.com": 0.0, ' \
                    '"P_emaildomain_suddenlink.net": 0.0, "P_emaildomain_twc.com": 0.0, ' \
                    '"P_emaildomain_verizon.net": 0.0, "P_emaildomain_web.de": 0.0, ' \
                    '"P_emaildomain_windstream.net": 0.0, "P_emaildomain_yahoo.co.jp": 0.0, ' \
                    '"P_emaildomain_yahoo.co.uk": 0.0, "P_emaildomain_yahoo.com": 1.0, ' \
                    '"P_emaildomain_yahoo.com.mx": 0.0, "P_emaildomain_yahoo.de": 0.0, ' \
                    '"P_emaildomain_yahoo.es": 0.0, "P_emaildomain_yahoo.fr": 0.0, ' \
                    '"P_emaildomain_ymail.com": 0.0, "R_emaildomain_aim.com": 0.0, ' \
                    '"R_emaildomain_anonymous.com": 0.0, "R_emaildomain_aol.com": 0.0, ' \
                    '"R_emaildomain_att.net": 0.0, "R_emaildomain_bellsouth.net": 0.0, ' \
                    '"R_emaildomain_cableone.net": 0.0, "R_emaildomain_centurylink.net": 0.0, ' \
                    '"R_emaildomain_cfl.rr.com": 0.0, "R_emaildomain_charter.net": 0.0, ' \
                    '"R_emaildomain_comcast.net": 0.0, "R_emaildomain_cox.net": 0.0, ' \
                    '"R_emaildomain_earthlink.net": 0.0, "R_emaildomain_embarqmail.com": 0.0, ' \
                    '"R_emaildomain_frontier.com": 0.0, "R_emaildomain_frontiernet.net": 0.0, ' \
                    '"R_emaildomain_gmail": 0.0, "R_emaildomain_gmail.com": 0.0, "R_emaildomain_gmx.de": 0.0, ' \
                    '"R_emaildomain_hotmail.co.uk": 0.0, "R_emaildomain_hotmail.com": 0.0, ' \
                    '"R_emaildomain_hotmail.de": 0.0, "R_emaildomain_hotmail.es": 0.0, ' \
                    '"R_emaildomain_hotmail.fr": 0.0, "R_emaildomain_icloud.com": 0.0, ' \
                    '"R_emaildomain_juno.com": 0.0, "R_emaildomain_live.com": 0.0, ' \
                    '"R_emaildomain_live.com.mx": 0.0, "R_emaildomain_live.fr": 0.0, ' \
                    '"R_emaildomain_mac.com": 0.0, "R_emaildomain_mail.com": 0.0, ' \
                    '"R_emaildomain_me.com": 0.0, "R_emaildomain_msn.com": 0.0, ' \
                    '"R_emaildomain_netzero.com": 0.0, "R_emaildomain_netzero.net": 0.0, ' \
                    '"R_emaildomain_optonline.net": 0.0, "R_emaildomain_outlook.com": 0.0, ' \
                    '"R_emaildomain_outlook.es": 0.0, "R_emaildomain_prodigy.net.mx": 0.0, ' \
                    '"R_emaildomain_protonmail.com": 0.0, "R_emaildomain_ptd.net": 0.0, ' \
                    '"R_emaildomain_q.com": 0.0, "R_emaildomain_roadrunner.com": 0.0, ' \
                    '"R_emaildomain_rocketmail.com": 0.0, "R_emaildomain_sbcglobal.net": 0.0, ' \
                    '"R_emaildomain_sc.rr.com": 0.0, "R_emaildomain_scranton.edu": 0.0, ' \
                    '"R_emaildomain_servicios-ta.com": 0.0, "R_emaildomain_suddenlink.net": 0.0, ' \
                    '"R_emaildomain_twc.com": 0.0, "R_emaildomain_verizon.net": 0.0, "R_emaildomain_web.de": 0.0, ' \
                    '"R_emaildomain_windstream.net": 0.0, "R_emaildomain_yahoo.co.jp": 0.0, ' \
                    '"R_emaildomain_yahoo.co.uk": 0.0, "R_emaildomain_yahoo.com": 0.0, ' \
                    '"R_emaildomain_yahoo.com.mx": 0.0, "R_emaildomain_yahoo.de": 0.0, ' \
                    '"R_emaildomain_yahoo.es": 0.0, "R_emaildomain_yahoo.fr": 0.0, ' \
                    '"R_emaildomain_ymail.com": 0.0, "M1_F": 0.0, "M1_T": 1.0, "M2_F": 1.0, ' \
                    '"M2_T": 0.0, "M3_F": 1.0, "M3_T": 0.0, "M4_M0": 0.0, "M4_M1": 0.0, "M4_M2": 0.0, "M5_F": 0.0, ' \
                    '"M5_T": 0.0, "M6_F": 1.0, "M6_T": 0.0, "M7_F": 1.0, "M7_T": 0.0, "M8_F": 1.0, "M8_T": 0.0, ' \
                    '"M9_F": 1.0, "M9_T": 0.0, "DeviceType_desktop": 0.0, "DeviceType_mobile": 0.0, ' \
                    '"id_12_Found": 0.0, "id_12_NotFound": 0.0, "id_15_Found": 0.0, "id_15_New": 0.0, ' \
                    '"id_15_Unknown": 0.0, "id_16_Found": 0.0, "id_16_NotFound": 0.0, "id_28_Found": 0.0, ' \
                    '"id_28_New": 0.0, "id_29_Found": 0.0, "id_29_NotFound": 0.0, "id_35_F": 0.0, ' \
                    '"id_35_T": 0.0, "id_36_F": 0.0, "id_36_T": 0.0, "id_37_F": 0.0, "id_37_T": 0.0, ' \
                    '"id_38_F": 0.0, "id_38_T": 0.0}'


@patch('fastapi_service.predictor.get_prep_data_pipe_from_pkl')
@patch('fastapi_service.predictor.get_model')
def test_prep_data(mock_get_model, mock_get_prep_data_pipe_from_pkl):
    dict_for_df = defaultdict(list)
    prepared_transaction_dict = json.loads(prepared_trn_json)
    for k, v in prepared_transaction_dict.items():
        dict_for_df[k].append(v)
    df_prepared = pd.DataFrame(dict_for_df)
    df_raw = pd.DataFrame()
    prep_data_pipe_mock = Mock()
    prep_data_pipe_mock.transform = Mock(return_value=df_prepared)
    mock_get_prep_data_pipe_from_pkl.return_value = prep_data_pipe_mock
    mock_get_model.return_value = Mock()

    predictor = Predictor(clf_name='logreg')
    actual = predictor.prep_data(df_raw)

    assert df_prepared.equals(actual)
    prep_data_pipe_mock.transform.assert_called_once_with(df_raw)


@patch('fastapi_service.predictor.get_prep_data_pipe_from_pkl')
@patch('fastapi_service.predictor.get_model')
def test_get_dataframe(mock_get_model, mock_get_prep_data_pipe_from_pkl):
    dict_for_df = defaultdict(list)
    raw_transaction_dict = json.loads(initial_trn_json)
    for k, v in raw_transaction_dict.items():
        dict_for_df[k].append(v)
    df_raw = pd.DataFrame(dict_for_df)
    test_data = [df_raw,
                 Transaction.parse_obj(json.loads(initial_trn_json)),
                 [Transaction.parse_obj(json.loads(initial_trn_json))],
                 dict_for_df]
    mock_get_prep_data_pipe_from_pkl.return_value = Mock()
    mock_get_model.return_value = Mock()

    predictor = Predictor(clf_name='logreg')
    for obj in test_data:
        actual = predictor._get_dataframe(obj)
        assert df_raw.equals(actual)


@patch('fastapi_service.predictor.get_prep_data_pipe_from_pkl')
@patch('fastapi_service.predictor.get_model')
@patch('fastapi_service.predictor.Predictor._get_dataframe')
def test_ml_predict(mock_get_dataframe, mock_get_model, mock_get_prep_data_pipe_from_pkl):
    dict_for_df = defaultdict(list)
    prepared_transaction_dict = json.loads(prepared_trn_json)
    for k, v in prepared_transaction_dict.items():
        dict_for_df[k].append(v)
    df_prepared = pd.DataFrame(dict_for_df)
    df_raw = pd.DataFrame()
    prep_data_pipe_mock = Mock()
    prep_data_pipe_mock.transform = Mock(return_value=df_prepared)
    mock_get_prep_data_pipe_from_pkl.return_value = prep_data_pipe_mock

    expected = np.ones(1, dtype=int)
    clf_mock = Mock()
    clf_mock.predict = Mock(return_value=expected)
    mock_get_model.return_value = clf_mock

    mock_get_dataframe.return_value = df_raw

    predictor = Predictor(clf_name='logreg')
    actual = predictor.predict(df_raw)

    assert expected == actual
    mock_get_dataframe.assert_called_once_with(df_raw)
    prep_data_pipe_mock.transform.assert_called_once_with(df_raw)
    mock_get_model.assert_called_once_with('logreg')
    clf_mock.predict.assert_called_once_with(df_prepared)


@patch('fastapi_service.predictor.get_prep_data_pipe_from_pkl')
@patch('fastapi_service.predictor.get_fcnn_model')
@patch('fastapi_service.predictor.Predictor._get_dataframe')
@patch('fastapi_service.predictor.nn_predict')
def test_dl_predict(mock_nn_predict, mock_get_dataframe, mock_get_fcnn_model, mock_get_prep_data_pipe_from_pkl):
    dict_for_df = defaultdict(list)
    prepared_transaction_dict = json.loads(prepared_trn_json)
    for k, v in prepared_transaction_dict.items():
        dict_for_df[k].append(v)
    df_prepared = pd.DataFrame(dict_for_df)
    df_raw = pd.DataFrame()
    prep_data_pipe_mock = Mock()
    prep_data_pipe_mock.transform = Mock(return_value=df_prepared)
    mock_get_prep_data_pipe_from_pkl.return_value = prep_data_pipe_mock

    expected = np.ones(1, dtype=int)
    clf_mock = Mock()
    optimizer_mock = Mock()
    scheduler_mock = Mock()

    mock_get_fcnn_model.return_value = (clf_mock, optimizer_mock, scheduler_mock)
    mock_get_dataframe.return_value = df_raw
    mock_nn_predict.return_value = expected

    predictor = Predictor(clf_name='fcnn')
    actual = predictor.predict(df_raw)

    assert expected == actual
    mock_get_dataframe.assert_called_once_with(df_raw)
    prep_data_pipe_mock.transform.assert_called_once_with(df_raw)
    mock_get_fcnn_model.assert_called_once()
    mock_nn_predict.assert_called_once_with(clf_mock, df_prepared)


@patch('fastapi_service.predictor.get_prep_data_pipe_from_pkl')
@patch('fastapi_service.predictor.get_model')
@patch('fastapi_service.predictor.Predictor._get_dataframe')
def test_ml_fit(mock_get_dataframe, mock_get_model, mock_get_prep_data_pipe_from_pkl):
    dict_for_df = defaultdict(list)
    prepared_transaction_dict = json.loads(prepared_trn_json)
    for k, v in prepared_transaction_dict.items():
        dict_for_df[k].append(v)
    df_prepared = pd.DataFrame(dict_for_df)

    dict_for_df = defaultdict(list)
    raw_transaction_dict = json.loads(initial_trn_json)
    for k, v in raw_transaction_dict.items():
        dict_for_df[k].append(v)
    df_raw = pd.DataFrame(dict_for_df)

    prep_data_pipe_mock = Mock()
    prep_data_pipe_mock.transform = Mock(return_value=df_prepared)
    mock_get_prep_data_pipe_from_pkl.return_value = prep_data_pipe_mock

    y_train = df_raw['isFraud']

    clf_mock = Mock()
    clf_mock.fit = Mock()
    mock_get_model.return_value = clf_mock

    mock_get_dataframe.return_value = df_raw

    predictor = Predictor(clf_name='logreg')
    predictor.train(df_raw)

    mock_get_dataframe.assert_called_once_with(df_raw)
    prep_data_pipe_mock.transform.assert_called_once_with(df_raw)
    mock_get_model.assert_called_once_with('logreg')
    clf_mock.fit.assert_called_once_with(df_prepared, y_train)


@patch('fastapi_service.predictor.get_prep_data_pipe_from_pkl')
@patch('fastapi_service.predictor.get_fcnn_model')
@patch('fastapi_service.predictor.Predictor._get_dataframe')
@patch('fastapi_service.predictor.nn_train')
def test_dl_fit(mock_nn_train, mock_get_dataframe, mock_get_fcnn_model, mock_get_prep_data_pipe_from_pkl):
    dict_for_df = defaultdict(list)
    prepared_transaction_dict = json.loads(prepared_trn_json)
    for k, v in prepared_transaction_dict.items():
        dict_for_df[k].append(v)
    df_prepared = pd.DataFrame(dict_for_df)

    dict_for_df = defaultdict(list)
    raw_transaction_dict = json.loads(initial_trn_json)
    for k, v in raw_transaction_dict.items():
        dict_for_df[k].append(v)
    df_raw = pd.DataFrame(dict_for_df)
    y_train = df_raw['isFraud']

    prep_data_pipe_mock = Mock()
    prep_data_pipe_mock.transform = Mock(return_value=df_prepared)
    mock_get_prep_data_pipe_from_pkl.return_value = prep_data_pipe_mock

    clf_mock = Mock()
    optimizer_mock = Mock()
    scheduler_mock = Mock()

    mock_get_fcnn_model.return_value = (clf_mock, optimizer_mock, scheduler_mock)

    mock_get_dataframe.return_value = df_raw

    predictor = Predictor(clf_name='fcnn')
    predictor.train(df_raw)

    mock_get_dataframe.assert_called_once_with(df_raw)
    prep_data_pipe_mock.transform.assert_called_once_with(df_raw)
    mock_get_fcnn_model.assert_called_once()
    mock_nn_train.assert_called_once_with(clf_mock, optimizer_mock, scheduler_mock, df_prepared, y_train)
