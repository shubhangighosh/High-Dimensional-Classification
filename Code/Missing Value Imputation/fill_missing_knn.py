from fancyimpute import KNN
from pandas import read_csv
import pandas as pd
import numpy as np

lst = ["f_0","f_1","f_2","f_3","f_4","f_5","f_6","f_7","f_8","f_9",
"f_10","f_11","f_12","f_13","f_14","f_15","f_16","f_17","f_18","f_19",
"f_20","f_21","f_22","f_23","f_24","f_25","f_26","f_27","f_28","f_29",
"f_30","f_31","f_32","f_33","f_34","f_35","f_36","f_37","f_38","f_39",
"f_40","f_41","f_42","f_43","f_44","f_45","f_46","f_47","f_48","f_49",
"f_50","f_51","f_52","f_53","f_54","f_55","f_56","f_57","f_58","f_59",
"f_60","f_61","f_62","f_63","f_64","f_65","f_66","f_67","f_68","f_69",
"f_70","f_71","f_72","f_73","f_74","f_75","f_76","f_77","f_78","f_79",
"f_80","f_81","f_82","f_83","f_84","f_85","f_86","f_87","f_88","f_89",
"f_90","f_91","f_92","f_93","f_94","f_95","f_96","f_97","f_98","f_99",
"f_100","f_101","f_102","f_103","f_104","f_105","f_106","f_107","f_108",
"f_109","f_110","f_111","f_112","f_113","f_114","f_115","f_116","f_117",
"f_118","f_119","f_120","f_121","f_122","f_123","f_124","f_125","f_126",
"f_127","f_128","f_129","f_130","f_131","f_132","f_133","f_134","f_135",
"f_136","f_137","f_138","f_139","f_140","f_141","f_142","f_143","f_144",
"f_145","f_146","f_147","f_148","f_149","f_150","f_151","f_152","f_153",
"f_154","f_155","f_156","f_157","f_158","f_159","f_160","f_161","f_162",
"f_163","f_164","f_165","f_166","f_167","f_168","f_169","f_170","f_171",
"f_172","f_173","f_174","f_175","f_176","f_177","f_178","f_179","f_180",
"f_181","f_182","f_183","f_184","f_185","f_186","f_187","f_188","f_189",
"f_190","f_191","f_192","f_193","f_194","f_195","f_196","f_197","f_198",
"f_199","f_200","f_201","f_202","f_203","f_204","f_205","f_206","f_207",
"f_208","f_209","f_210","f_211","f_212","f_213","f_214","f_215","f_216",
"f_217","f_218","f_219","f_220","f_221","f_222","f_223","f_224","f_225",
"f_226","f_227","f_228","f_229","f_230","f_231","f_232","f_233","f_234",
"f_235","f_236","f_237","f_238","f_239","f_240","f_241","f_242","f_243",
"f_244","f_245","f_246","f_247","f_248","f_249","f_250","f_251","f_252",
"f_253","f_254","f_255","f_256","f_257","f_258","f_259","f_260","f_261",
"f_262","f_263","f_264","f_265","f_266","f_267","f_268","f_269","f_270",
"f_271","f_272","f_273","f_274","f_275","f_276","f_277","f_278","f_279",
"f_280","f_281","f_282","f_283","f_284","f_285","f_286","f_287","f_288",
"f_289","f_290","f_291","f_292","f_293","f_294","f_295","f_296","f_297",
"f_298","f_299","f_300","f_301","f_302","f_303","f_304","f_305","f_306",
"f_307","f_308","f_309","f_310","f_311","f_312","f_313","f_314","f_315",
"f_316","f_317","f_318","f_319","f_320","f_321","f_322","f_323","f_324",
"f_325","f_326","f_327","f_328","f_329","f_330","f_331","f_332","f_333",
"f_334","f_335","f_336","f_337","f_338","f_339","f_340","f_341","f_342",
"f_343","f_344","f_345","f_346","f_347","f_348","f_349","f_350","f_351",
"f_352","f_353","f_354","f_355","f_356","f_357","f_358","f_359","f_360",
"f_361","f_362","f_363","f_364","f_365","f_366","f_367","f_368","f_369",
"f_370","f_371","f_372","f_373","f_374","f_375","f_376","f_377","f_378",
"f_379","f_380","f_381","f_382","f_383","f_384","f_385","f_386","f_387",
"f_388","f_389","f_390","f_391","f_392","f_393","f_394","f_395","f_396",
"f_397","f_398","f_399","f_400","f_401","f_402","f_403","f_404","f_405","f_406","f_407","f_408","f_409","f_410","f_411","f_412","f_413","f_414","f_415","f_416","f_417","f_418","f_419","f_420","f_421","f_422","f_423","f_424","f_425","f_426","f_427","f_428","f_429","f_430","f_431","f_432","f_433","f_434","f_435","f_436","f_437","f_438","f_439","f_440","f_441","f_442","f_443","f_444","f_445","f_446","f_447","f_448","f_449","f_450","f_451","f_452","f_453","f_454","f_455","f_456","f_457","f_458","f_459","f_460","f_461","f_462","f_463","f_464","f_465","f_466","f_467","f_468","f_469","f_470","f_471","f_472","f_473","f_474","f_475","f_476","f_477","f_478","f_479","f_480","f_481","f_482","f_483","f_484","f_485","f_486","f_487","f_488","f_489","f_490","f_491","f_492","f_493","f_494","f_495","f_496","f_497","f_498","f_499","f_500","f_501","f_502","f_503","f_504","f_505","f_506","f_507","f_508","f_509","f_510","f_511","f_512","f_513","f_514","f_515","f_516","f_517","f_518","f_519","f_520","f_521","f_522","f_523","f_524","f_525","f_526","f_527","f_528","f_529","f_530","f_531","f_532","f_533","f_534","f_535","f_536","f_537","f_538","f_539","f_540","f_541","f_542","f_543","f_544","f_545","f_546","f_547","f_548","f_549","f_550","f_551","f_552","f_553","f_554","f_555","f_556","f_557","f_558","f_559","f_560","f_561","f_562","f_563","f_564","f_565","f_566","f_567","f_568","f_569","f_570","f_571","f_572","f_573","f_574","f_575","f_576","f_577","f_578","f_579","f_580","f_581","f_582","f_583","f_584","f_585","f_586","f_587","f_588","f_589","f_590","f_591","f_592","f_593","f_594","f_595","f_596","f_597","f_598","f_599","f_600","f_601","f_602","f_603","f_604","f_605","f_606","f_607","f_608","f_609","f_610","f_611","f_612","f_613","f_614","f_615","f_616","f_617","f_618","f_619","f_620","f_621","f_622","f_623","f_624","f_625","f_626","f_627","f_628","f_629","f_630","f_631","f_632","f_633","f_634","f_635","f_636","f_637","f_638","f_639","f_640","f_641","f_642","f_643","f_644","f_645","f_646","f_647","f_648","f_649","f_650","f_651","f_652","f_653","f_654","f_655","f_656","f_657","f_658","f_659","f_660","f_661","f_662","f_663","f_664","f_665","f_666","f_667","f_668","f_669","f_670","f_671","f_672","f_673","f_674","f_675","f_676","f_677","f_678","f_679","f_680","f_681","f_682","f_683","f_684","f_685","f_686","f_687","f_688","f_689","f_690","f_691","f_692","f_693","f_694","f_695","f_696","f_697","f_698","f_699","f_700","f_701","f_702","f_703","f_704","f_705","f_706","f_707","f_708","f_709","f_710","f_711","f_712","f_713","f_714","f_715","f_716","f_717","f_718","f_719","f_720","f_721","f_722","f_723","f_724","f_725","f_726","f_727","f_728","f_729","f_730","f_731","f_732","f_733","f_734","f_735","f_736","f_737","f_738","f_739","f_740","f_741","f_742","f_743","f_744","f_745","f_746","f_747","f_748","f_749","f_750","f_751","f_752","f_753","f_754","f_755","f_756","f_757","f_758","f_759","f_760","f_761","f_762","f_763","f_764","f_765","f_766","f_767","f_768","f_769","f_770","f_771","f_772","f_773","f_774","f_775","f_776","f_777","f_778","f_779","f_780","f_781","f_782","f_783","f_784","f_785","f_786","f_787","f_788","f_789","f_790","f_791","f_792","f_793","f_794","f_795","f_796","f_797","f_798","f_799","f_800","f_801","f_802","f_803","f_804","f_805","f_806","f_807","f_808","f_809","f_810","f_811","f_812","f_813","f_814","f_815","f_816","f_817","f_818","f_819","f_820","f_821","f_822","f_823","f_824","f_825","f_826","f_827","f_828","f_829","f_830","f_831","f_832","f_833","f_834","f_835","f_836","f_837","f_838","f_839","f_840","f_841","f_842","f_843","f_844","f_845","f_846","f_847","f_848","f_849","f_850","f_851","f_852","f_853","f_854","f_855","f_856","f_857","f_858","f_859","f_860","f_861","f_862","f_863","f_864","f_865","f_866","f_867","f_868","f_869","f_870","f_871","f_872","f_873","f_874","f_875","f_876","f_877","f_878","f_879","f_880","f_881","f_882","f_883","f_884","f_885","f_886","f_887","f_888","f_889","f_890","f_891","f_892","f_893","f_894","f_895","f_896","f_897","f_898","f_899","f_900","f_901","f_902","f_903","f_904","f_905","f_906","f_907","f_908","f_909","f_910","f_911","f_912","f_913","f_914","f_915","f_916","f_917","f_918","f_919","f_920","f_921","f_922","f_923","f_924","f_925","f_926","f_927","f_928","f_929","f_930","f_931","f_932","f_933","f_934","f_935","f_936","f_937","f_938","f_939","f_940","f_941","f_942","f_943","f_944","f_945","f_946","f_947","f_948","f_949","f_950","f_951","f_952","f_953","f_954","f_955","f_956","f_957","f_958","f_959","f_960","f_961","f_962","f_963","f_964","f_965","f_966","f_967","f_968","f_969","f_970","f_971","f_972","f_973","f_974","f_975","f_976","f_977","f_978","f_979","f_980","f_981","f_982","f_983","f_984","f_985","f_986","f_987","f_988","f_989","f_990","f_991","f_992","f_993","f_994","f_995","f_996","f_997","f_998","f_999","f_1000","f_1001","f_1002","f_1003","f_1004","f_1005","f_1006","f_1007","f_1008","f_1009","f_1010","f_1011","f_1012","f_1013","f_1014","f_1015","f_1016","f_1017","f_1018","f_1019","f_1020","f_1021","f_1022","f_1023","f_1024","f_1025","f_1026","f_1027","f_1028","f_1029","f_1030","f_1031","f_1032","f_1033","f_1034","f_1035","f_1036","f_1037","f_1038","f_1039","f_1040","f_1041","f_1042","f_1043","f_1044","f_1045","f_1046","f_1047","f_1048","f_1049","f_1050","f_1051","f_1052","f_1053","f_1054","f_1055","f_1056","f_1057","f_1058","f_1059","f_1060","f_1061","f_1062","f_1063","f_1064","f_1065","f_1066","f_1067","f_1068","f_1069","f_1070","f_1071","f_1072","f_1073","f_1074","f_1075","f_1076","f_1077","f_1078","f_1079","f_1080","f_1081","f_1082","f_1083","f_1084","f_1085","f_1086","f_1087","f_1088","f_1089","f_1090","f_1091","f_1092","f_1093","f_1094","f_1095","f_1096","f_1097","f_1098","f_1099","f_1100","f_1101","f_1102","f_1103","f_1104","f_1105","f_1106","f_1107","f_1108","f_1109","f_1110","f_1111","f_1112","f_1113","f_1114","f_1115","f_1116","f_1117","f_1118","f_1119","f_1120","f_1121","f_1122","f_1123","f_1124","f_1125","f_1126","f_1127","f_1128","f_1129","f_1130","f_1131","f_1132","f_1133","f_1134","f_1135","f_1136","f_1137","f_1138","f_1139","f_1140","f_1141","f_1142","f_1143","f_1144","f_1145","f_1146","f_1147","f_1148","f_1149","f_1150","f_1151","f_1152","f_1153","f_1154","f_1155","f_1156","f_1157","f_1158","f_1159","f_1160","f_1161","f_1162","f_1163","f_1164","f_1165","f_1166","f_1167","f_1168","f_1169","f_1170","f_1171","f_1172","f_1173","f_1174","f_1175","f_1176","f_1177","f_1178","f_1179","f_1180","f_1181","f_1182","f_1183","f_1184","f_1185","f_1186","f_1187","f_1188","f_1189","f_1190","f_1191","f_1192","f_1193","f_1194","f_1195","f_1196","f_1197","f_1198","f_1199","f_1200","f_1201","f_1202","f_1203","f_1204","f_1205","f_1206","f_1207","f_1208","f_1209","f_1210","f_1211","f_1212","f_1213","f_1214","f_1215","f_1216","f_1217","f_1218","f_1219","f_1220","f_1221","f_1222","f_1223","f_1224","f_1225","f_1226","f_1227","f_1228","f_1229","f_1230","f_1231","f_1232","f_1233","f_1234","f_1235","f_1236","f_1237","f_1238","f_1239","f_1240","f_1241","f_1242","f_1243","f_1244","f_1245","f_1246","f_1247","f_1248","f_1249","f_1250","f_1251","f_1252","f_1253","f_1254","f_1255","f_1256","f_1257","f_1258","f_1259","f_1260","f_1261","f_1262","f_1263","f_1264","f_1265","f_1266","f_1267","f_1268","f_1269","f_1270","f_1271","f_1272","f_1273","f_1274","f_1275","f_1276","f_1277","f_1278","f_1279","f_1280","f_1281","f_1282","f_1283","f_1284","f_1285","f_1286","f_1287","f_1288","f_1289","f_1290","f_1291","f_1292","f_1293","f_1294","f_1295","f_1296","f_1297","f_1298","f_1299","f_1300","f_1301","f_1302","f_1303","f_1304","f_1305","f_1306","f_1307","f_1308","f_1309","f_1310","f_1311","f_1312","f_1313","f_1314","f_1315","f_1316","f_1317","f_1318","f_1319","f_1320","f_1321","f_1322","f_1323","f_1324","f_1325","f_1326","f_1327","f_1328","f_1329","f_1330","f_1331","f_1332","f_1333","f_1334","f_1335","f_1336","f_1337","f_1338","f_1339","f_1340","f_1341","f_1342","f_1343","f_1344","f_1345","f_1346","f_1347","f_1348","f_1349","f_1350","f_1351","f_1352","f_1353","f_1354","f_1355","f_1356","f_1357","f_1358","f_1359","f_1360","f_1361","f_1362","f_1363","f_1364","f_1365","f_1366","f_1367","f_1368","f_1369","f_1370","f_1371","f_1372","f_1373","f_1374","f_1375","f_1376","f_1377","f_1378","f_1379","f_1380","f_1381","f_1382","f_1383","f_1384","f_1385","f_1386","f_1387","f_1388","f_1389","f_1390","f_1391","f_1392","f_1393","f_1394","f_1395","f_1396","f_1397","f_1398","f_1399","f_1400","f_1401","f_1402","f_1403","f_1404","f_1405","f_1406","f_1407","f_1408","f_1409","f_1410","f_1411","f_1412","f_1413","f_1414","f_1415","f_1416","f_1417","f_1418","f_1419","f_1420","f_1421","f_1422","f_1423","f_1424","f_1425","f_1426","f_1427","f_1428","f_1429","f_1430","f_1431","f_1432","f_1433","f_1434","f_1435","f_1436","f_1437","f_1438","f_1439","f_1440","f_1441","f_1442","f_1443","f_1444","f_1445","f_1446","f_1447","f_1448","f_1449","f_1450","f_1451","f_1452","f_1453","f_1454","f_1455","f_1456","f_1457","f_1458","f_1459","f_1460","f_1461","f_1462","f_1463","f_1464","f_1465","f_1466","f_1467","f_1468","f_1469","f_1470","f_1471","f_1472","f_1473","f_1474","f_1475","f_1476","f_1477","f_1478","f_1479","f_1480","f_1481","f_1482","f_1483","f_1484","f_1485","f_1486","f_1487","f_1488","f_1489","f_1490","f_1491","f_1492","f_1493","f_1494","f_1495","f_1496","f_1497","f_1498","f_1499","f_1500","f_1501","f_1502","f_1503","f_1504","f_1505","f_1506","f_1507","f_1508","f_1509","f_1510","f_1511","f_1512","f_1513","f_1514","f_1515","f_1516","f_1517","f_1518","f_1519","f_1520","f_1521","f_1522","f_1523","f_1524","f_1525","f_1526","f_1527","f_1528","f_1529","f_1530","f_1531","f_1532","f_1533","f_1534","f_1535","f_1536","f_1537","f_1538","f_1539","f_1540","f_1541","f_1542","f_1543","f_1544","f_1545","f_1546","f_1547","f_1548","f_1549","f_1550","f_1551","f_1552","f_1553","f_1554","f_1555","f_1556","f_1557","f_1558","f_1559","f_1560","f_1561","f_1562","f_1563","f_1564","f_1565","f_1566","f_1567","f_1568","f_1569","f_1570","f_1571","f_1572","f_1573","f_1574","f_1575","f_1576","f_1577","f_1578","f_1579","f_1580","f_1581","f_1582","f_1583","f_1584","f_1585","f_1586","f_1587","f_1588","f_1589","f_1590","f_1591","f_1592","f_1593","f_1594","f_1595","f_1596","f_1597","f_1598","f_1599","f_1600","f_1601","f_1602","f_1603","f_1604","f_1605","f_1606","f_1607","f_1608","f_1609","f_1610","f_1611","f_1612","f_1613","f_1614","f_1615","f_1616","f_1617","f_1618","f_1619","f_1620","f_1621","f_1622","f_1623","f_1624","f_1625","f_1626","f_1627","f_1628","f_1629","f_1630","f_1631","f_1632","f_1633","f_1634","f_1635","f_1636","f_1637","f_1638","f_1639","f_1640","f_1641","f_1642","f_1643","f_1644","f_1645","f_1646","f_1647","f_1648","f_1649","f_1650","f_1651","f_1652","f_1653","f_1654","f_1655","f_1656","f_1657","f_1658","f_1659","f_1660","f_1661","f_1662","f_1663","f_1664","f_1665","f_1666","f_1667","f_1668","f_1669","f_1670","f_1671","f_1672","f_1673","f_1674","f_1675","f_1676","f_1677","f_1678","f_1679","f_1680","f_1681","f_1682","f_1683","f_1684","f_1685","f_1686","f_1687","f_1688","f_1689","f_1690","f_1691","f_1692","f_1693","f_1694","f_1695","f_1696","f_1697","f_1698","f_1699","f_1700","f_1701","f_1702","f_1703","f_1704","f_1705","f_1706","f_1707","f_1708","f_1709","f_1710","f_1711","f_1712","f_1713","f_1714","f_1715","f_1716","f_1717","f_1718","f_1719","f_1720","f_1721","f_1722","f_1723","f_1724","f_1725","f_1726","f_1727","f_1728","f_1729","f_1730","f_1731","f_1732","f_1733","f_1734","f_1735","f_1736","f_1737","f_1738","f_1739","f_1740","f_1741","f_1742","f_1743","f_1744","f_1745","f_1746","f_1747","f_1748","f_1749","f_1750","f_1751","f_1752","f_1753","f_1754","f_1755","f_1756","f_1757","f_1758","f_1759","f_1760","f_1761","f_1762","f_1763","f_1764","f_1765","f_1766","f_1767","f_1768","f_1769","f_1770","f_1771","f_1772","f_1773","f_1774","f_1775","f_1776","f_1777","f_1778","f_1779","f_1780","f_1781","f_1782","f_1783","f_1784","f_1785","f_1786","f_1787","f_1788","f_1789","f_1790","f_1791","f_1792","f_1793","f_1794","f_1795","f_1796","f_1797","f_1798","f_1799","f_1800","f_1801","f_1802","f_1803","f_1804","f_1805","f_1806","f_1807","f_1808","f_1809","f_1810","f_1811","f_1812","f_1813","f_1814","f_1815","f_1816","f_1817","f_1818","f_1819","f_1820","f_1821","f_1822","f_1823","f_1824","f_1825","f_1826","f_1827","f_1828","f_1829","f_1830","f_1831","f_1832","f_1833","f_1834","f_1835","f_1836","f_1837","f_1838","f_1839","f_1840","f_1841","f_1842","f_1843","f_1844","f_1845","f_1846","f_1847","f_1848","f_1849","f_1850","f_1851","f_1852","f_1853","f_1854","f_1855","f_1856","f_1857","f_1858","f_1859","f_1860","f_1861","f_1862","f_1863","f_1864","f_1865","f_1866","f_1867","f_1868","f_1869","f_1870","f_1871","f_1872","f_1873","f_1874","f_1875","f_1876","f_1877","f_1878","f_1879","f_1880","f_1881","f_1882","f_1883","f_1884","f_1885","f_1886","f_1887","f_1888","f_1889","f_1890","f_1891","f_1892","f_1893","f_1894","f_1895","f_1896","f_1897","f_1898","f_1899","f_1900","f_1901","f_1902","f_1903","f_1904","f_1905","f_1906","f_1907","f_1908","f_1909","f_1910","f_1911","f_1912","f_1913","f_1914","f_1915","f_1916","f_1917","f_1918","f_1919","f_1920","f_1921","f_1922","f_1923","f_1924","f_1925","f_1926","f_1927","f_1928","f_1929","f_1930","f_1931","f_1932","f_1933","f_1934","f_1935","f_1936","f_1937","f_1938","f_1939","f_1940","f_1941","f_1942","f_1943","f_1944","f_1945","f_1946","f_1947","f_1948","f_1949","f_1950","f_1951","f_1952","f_1953","f_1954","f_1955","f_1956","f_1957","f_1958","f_1959","f_1960","f_1961","f_1962","f_1963","f_1964","f_1965","f_1966","f_1967","f_1968","f_1969","f_1970","f_1971","f_1972","f_1973","f_1974","f_1975","f_1976","f_1977","f_1978","f_1979","f_1980","f_1981","f_1982","f_1983","f_1984","f_1985","f_1986","f_1987","f_1988","f_1989","f_1990","f_1991","f_1992","f_1993","f_1994","f_1995","f_1996","f_1997","f_1998","f_1999","f_2000","f_2001","f_2002","f_2003","f_2004","f_2005","f_2006","f_2007","f_2008","f_2009","f_2010","f_2011","f_2012","f_2013","f_2014","f_2015","f_2016","f_2017","f_2018","f_2019","f_2020","f_2021","f_2022","f_2023","f_2024","f_2025","f_2026","f_2027","f_2028","f_2029","f_2030","f_2031","f_2032","f_2033","f_2034","f_2035","f_2036","f_2037","f_2038","f_2039","f_2040","f_2041","f_2042","f_2043","f_2044","f_2045","f_2046","f_2047","f_2048","f_2049","f_2050","f_2051","f_2052","f_2053","f_2054","f_2055","f_2056","f_2057","f_2058","f_2059","f_2060","f_2061","f_2062","f_2063","f_2064","f_2065","f_2066","f_2067","f_2068","f_2069","f_2070","f_2071","f_2072","f_2073","f_2074","f_2075","f_2076","f_2077","f_2078","f_2079","f_2080","f_2081","f_2082","f_2083","f_2084","f_2085","f_2086","f_2087","f_2088","f_2089","f_2090","f_2091","f_2092","f_2093","f_2094","f_2095","f_2096","f_2097","f_2098","f_2099","f_2100","f_2101","f_2102","f_2103","f_2104","f_2105","f_2106","f_2107","f_2108","f_2109","f_2110","f_2111","f_2112","f_2113","f_2114","f_2115","f_2116","f_2117","f_2118","f_2119","f_2120","f_2121","f_2122","f_2123","f_2124","f_2125","f_2126","f_2127","f_2128","f_2129","f_2130","f_2131","f_2132","f_2133","f_2134","f_2135","f_2136","f_2137","f_2138","f_2139","f_2140","f_2141","f_2142","f_2143","f_2144","f_2145","f_2146","f_2147","f_2148","f_2149","f_2150","f_2151","f_2152","f_2153","f_2154","f_2155","f_2156","f_2157","f_2158","f_2159","f_2160","f_2161","f_2162","f_2163","f_2164","f_2165","f_2166","f_2167","f_2168","f_2169","f_2170","f_2171","f_2172","f_2173","f_2174","f_2175","f_2176","f_2177","f_2178","f_2179","f_2180","f_2181","f_2182","f_2183","f_2184","f_2185","f_2186","f_2187","f_2188","f_2189","f_2190","f_2191","f_2192","f_2193","f_2194","f_2195","f_2196","f_2197","f_2198","f_2199","f_2200","f_2201","f_2202","f_2203","f_2204","f_2205","f_2206","f_2207","f_2208","f_2209","f_2210","f_2211","f_2212","f_2213","f_2214","f_2215","f_2216","f_2217","f_2218","f_2219","f_2220","f_2221","f_2222","f_2223","f_2224","f_2225","f_2226","f_2227","f_2228","f_2229","f_2230","f_2231","f_2232","f_2233","f_2234","f_2235","f_2236","f_2237","f_2238","f_2239","f_2240","f_2241","f_2242","f_2243","f_2244","f_2245","f_2246","f_2247","f_2248","f_2249","f_2250","f_2251","f_2252","f_2253","f_2254","f_2255","f_2256","f_2257","f_2258","f_2259","f_2260","f_2261","f_2262","f_2263","f_2264","f_2265","f_2266","f_2267","f_2268","f_2269","f_2270","f_2271","f_2272","f_2273","f_2274","f_2275","f_2276","f_2277","f_2278","f_2279","f_2280","f_2281","f_2282","f_2283","f_2284","f_2285","f_2286","f_2287","f_2288","f_2289","f_2290","f_2291","f_2292","f_2293","f_2294","f_2295","f_2296","f_2297","f_2298","f_2299","f_2300","f_2301","f_2302","f_2303","f_2304","f_2305","f_2306","f_2307","f_2308","f_2309","f_2310","f_2311","f_2312","f_2313","f_2314","f_2315","f_2316","f_2317","f_2318","f_2319","f_2320","f_2321","f_2322","f_2323","f_2324","f_2325","f_2326","f_2327","f_2328","f_2329","f_2330","f_2331","f_2332","f_2333","f_2334","f_2335","f_2336","f_2337","f_2338","f_2339","f_2340","f_2341","f_2342","f_2343","f_2344","f_2345","f_2346","f_2347","f_2348","f_2349","f_2350","f_2351","f_2352","f_2353","f_2354","f_2355","f_2356","f_2357","f_2358","f_2359","f_2360","f_2361","f_2362","f_2363","f_2364","f_2365","f_2366","f_2367","f_2368","f_2369","f_2370","f_2371","f_2372","f_2373","f_2374","f_2375","f_2376","f_2377","f_2378","f_2379","f_2380","f_2381","f_2382","f_2383","f_2384","f_2385","f_2386","f_2387","f_2388","f_2389","f_2390","f_2391","f_2392","f_2393","f_2394","f_2395","f_2396","f_2397","f_2398","f_2399","f_2400","f_2401","f_2402","f_2403","f_2404","f_2405","f_2406","f_2407","f_2408","f_2409","f_2410","f_2411","f_2412","f_2413","f_2414","f_2415","f_2416","f_2417","f_2418","f_2419","f_2420","f_2421","f_2422","f_2423","f_2424","f_2425","f_2426","f_2427","f_2428","f_2429","f_2430","f_2431","f_2432","f_2433","f_2434","f_2435","f_2436","f_2437","f_2438","f_2439","f_2440","f_2441","f_2442","f_2443","f_2444","f_2445","f_2446","f_2447","f_2448","f_2449","f_2450","f_2451","f_2452","f_2453","f_2454","f_2455","f_2456","f_2457","f_2458","f_2459","f_2460","f_2461","f_2462","f_2463","f_2464","f_2465","f_2466","f_2467","f_2468","f_2469","f_2470","f_2471","f_2472","f_2473","f_2474","f_2475","f_2476","f_2477","f_2478","f_2479","f_2480","f_2481","f_2482","f_2483","f_2484","f_2485","f_2486","f_2487","f_2488","f_2489","f_2490","f_2491","f_2492","f_2493","f_2494","f_2495","f_2496","f_2497","f_2498","f_2499","f_2500","f_2501","f_2502","f_2503","f_2504","f_2505","f_2506","f_2507","f_2508","f_2509","f_2510","f_2511","f_2512","f_2513","f_2514","f_2515","f_2516","f_2517","f_2518","f_2519","f_2520","f_2521","f_2522","f_2523","f_2524","f_2525","f_2526","f_2527","f_2528","f_2529","f_2530","f_2531","f_2532","f_2533","f_2534","f_2535","f_2536","f_2537","f_2538","f_2539","f_2540","f_2541","f_2542","f_2543","f_2544","f_2545","f_2546","f_2547","f_2548","f_2549","f_2550","f_2551","f_2552","f_2553","f_2554","f_2555","f_2556","f_2557","f_2558","f_2559","f_2560","f_2561","f_2562","f_2563","f_2564","f_2565","f_2566","f_2567","f_2568","f_2569","f_2570","f_2571","f_2572","f_2573","f_2574","f_2575","f_2576","f_2577","f_2578","f_2579","f_2580","f_2581","f_2582","f_2583","f_2584","f_2585","f_2586","f_2587","f_2588","f_2589","f_2590","f_2591","f_2592","f_2593","f_2594","f_2595","f_2596","f_2597","f_2598",
"f_2599","label"]

if True:
    data = read_csv(open('train.csv','r'),na_values='').as_matrix()
    X1 = data[:,1:-1] # input features
    Y1 = data[:,-1].astype('int') # input features

    X1 = KNN(k=11).complete(X1)

    train = np.concatenate((X1,np.reshape(Y1,(-1,1))),axis=1)
    pd.DataFrame(train).to_csv('train_knn11.csv', header = lst)


    print('Train done:',train.shape,data.shape)


    data = read_csv(open('test.csv','r'),na_values='').as_matrix()
    X2 = data[:,1:] # features
    train = X1.shape[0]

    X = np.concatenate((X1,X2))
    del X1,X2

    X_net = KNN(k=11).complete(X)
    del X

    test = X_net[train:]
    del X_net

    pd.DataFrame(test).to_csv('test_knn11.csv', header = lst[:-1])

    print('Test done:',test.shape,data.shape)
