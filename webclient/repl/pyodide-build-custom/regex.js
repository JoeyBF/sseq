var Module=typeof pyodide._module!=="undefined"?pyodide._module:{};Module.checkABI(1);if(!Module.expectedDataFileDownloads){Module.expectedDataFileDownloads=0;Module.finishedDataFileDownloads=0}Module.expectedDataFileDownloads++;(function(){var loadPackage=function(metadata){var PACKAGE_PATH;if(typeof window==="object"){PACKAGE_PATH=window["encodeURIComponent"](window.location.pathname.toString().substring(0,window.location.pathname.toString().lastIndexOf("/"))+"/")}else if(typeof location!=="undefined"){PACKAGE_PATH=encodeURIComponent(location.pathname.toString().substring(0,location.pathname.toString().lastIndexOf("/"))+"/")}else{throw"using preloaded data can only be done on a web page or in a web worker"}var PACKAGE_NAME="regex.data";var REMOTE_PACKAGE_BASE="regex.data";if(typeof Module["locateFilePackage"]==="function"&&!Module["locateFile"]){Module["locateFile"]=Module["locateFilePackage"];err("warning: you defined Module.locateFilePackage, that has been renamed to Module.locateFile (using your locateFilePackage for now)")}var REMOTE_PACKAGE_NAME=Module["locateFile"]?Module["locateFile"](REMOTE_PACKAGE_BASE,""):REMOTE_PACKAGE_BASE;var REMOTE_PACKAGE_SIZE=metadata.remote_package_size;var PACKAGE_UUID=metadata.package_uuid;function fetchRemotePackage(packageName,packageSize,callback,errback){var xhr=new XMLHttpRequest;xhr.open("GET",packageName,true);xhr.responseType="arraybuffer";xhr.onprogress=function(event){var url=packageName;var size=packageSize;if(event.total)size=event.total;if(event.loaded){if(!xhr.addedTotal){xhr.addedTotal=true;if(!Module.dataFileDownloads)Module.dataFileDownloads={};Module.dataFileDownloads[url]={loaded:event.loaded,total:size}}else{Module.dataFileDownloads[url].loaded=event.loaded}var total=0;var loaded=0;var num=0;for(var download in Module.dataFileDownloads){var data=Module.dataFileDownloads[download];total+=data.total;loaded+=data.loaded;num++}total=Math.ceil(total*Module.expectedDataFileDownloads/num);if(Module["setStatus"])Module["setStatus"]("Downloading data... ("+loaded+"/"+total+")")}else if(!Module.dataFileDownloads){if(Module["setStatus"])Module["setStatus"]("Downloading data...")}};xhr.onerror=function(event){throw new Error("NetworkError for: "+packageName)};xhr.onload=function(event){if(xhr.status==200||xhr.status==304||xhr.status==206||xhr.status==0&&xhr.response){var packageData=xhr.response;callback(packageData)}else{throw new Error(xhr.statusText+" : "+xhr.responseURL)}};xhr.send(null)}function handleError(error){console.error("package error:",error)}var fetchedCallback=null;var fetched=Module["getPreloadedPackage"]?Module["getPreloadedPackage"](REMOTE_PACKAGE_NAME,REMOTE_PACKAGE_SIZE):null;if(!fetched)fetchRemotePackage(REMOTE_PACKAGE_NAME,REMOTE_PACKAGE_SIZE,function(data){if(fetchedCallback){fetchedCallback(data);fetchedCallback=null}else{fetched=data}},handleError);function runWithFS(){function assert(check,msg){if(!check)throw msg+(new Error).stack}Module["FS_createPath"]("/","lib",true,true);Module["FS_createPath"]("/lib","python3.8",true,true);Module["FS_createPath"]("/lib/python3.8","site-packages",true,true);Module["FS_createPath"]("/lib/python3.8/site-packages","regex",true,true);function DataRequest(start,end,audio){this.start=start;this.end=end;this.audio=audio}DataRequest.prototype={requests:{},open:function(mode,name){this.name=name;this.requests[name]=this;Module["addRunDependency"]("fp "+this.name)},send:function(){},onload:function(){var byteArray=this.byteArray.subarray(this.start,this.end);this.finish(byteArray)},finish:function(byteArray){var that=this;Module["FS_createPreloadedFile"](this.name,null,byteArray,true,true,function(){Module["removeRunDependency"]("fp "+that.name)},function(){if(that.audio){Module["removeRunDependency"]("fp "+that.name)}else{err("Preloading file "+that.name+" failed")}},false,true);this.requests[this.name]=null}};function processPackageData(arrayBuffer){Module.finishedDataFileDownloads++;assert(arrayBuffer,"Loading data file failed.");assert(arrayBuffer instanceof ArrayBuffer,"bad input to processPackageData");var byteArray=new Uint8Array(arrayBuffer);var curr;var compressedData={data:null,cachedOffset:521650,cachedIndexes:[-1,-1],cachedChunks:[null,null],offsets:[0,1228,2260,3471,4632,5619,6597,7691,8492,9367,10010,10974,12114,13117,14247,15291,16440,17669,18532,19502,20594,21640,22693,23967,25119,26144,27479,28454,29696,30997,32018,32787,33839,34856,36248,37430,38701,40068,41316,42599,43467,44236,44897,45575,46419,47014,47791,48543,49417,49892,50643,51302,51920,52439,52989,53760,54554,55714,56578,57362,57998,58817,59349,59623,60230,60794,61588,62165,62797,63413,63998,64711,65258,65916,66789,67445,67954,68485,69113,69838,70689,71601,72187,72849,73641,74424,75190,75818,76634,77425,77962,78622,79367,80150,81185,82283,83297,84303,85289,86203,86959,87621,88229,89194,89724,90488,91128,92071,92746,93511,94138,94663,95368,96255,97e3,97866,98775,100076,100853,101513,102557,103087,103644,104503,105474,106185,106696,107883,108689,109596,110645,111522,112106,112753,113464,114038,114315,115583,116733,117112,117441,118299,119590,120765,122155,123397,124586,125848,126994,128063,129417,130615,131787,132798,133976,135437,136942,138465,139859,140999,142335,143783,144999,146102,146991,148292,149214,149727,150326,151260,152191,153264,154016,154982,155656,156740,157829,158927,160253,161320,162561,163748,164630,165924,167305,168358,169588,170866,171896,172424,173060,173715,174396,175111,175916,176924,177790,178799,179464,179684,180716,181917,183037,184246,185505,186249,187147,187935,189214,190518,191245,192244,193219,194313,195486,196612,197369,198607,199506,200220,201335,202405,203724,204310,204660,205137,205659,206128,206336,206387,206873,208071,208643,209102,209597,210159,210886,211860,213117,214111,215008,216133,217248,218489,219635,220885,222051,223413,224244,224819,226078,227050,227957,229383,230149,230486,231413,232238,233183,233901,234851,235634,236672,237541,238697,239538,240468,241081,241533,242181,242659,243194,244125,244416,245465,245838,246235,247111,248323,249623,250526,251346,252781,254188,255617,257173,258512,259974,260789,261903,263301,264895,266304,267204,268402,269441,270328,271598,272750,273796,275287,276600,277931,279198,280063,280893,281738,282723,283883,285358,286678,287729,288758,289802,290855,291914,292969,294020,295085,296148,297203,298266,299326,300389,301438,302500,303560,304613,305659,306721,307777,308829,309886,310330,311029,311608,312334,313097,314071,314785,314810,314835,315842,317700,319481,321256,323229,324558,326025,327782,328966,330354,332078,333342,334958,336365,337498,338837,339786,340862,342114,343531,344542,345811,346888,348267,349356,350387,351643,352623,353545,354792,356190,357237,358577,359585,360802,361535,363005,363954,365106,366451,367747,368389,369292,370059,371241,372401,373554,374657,376063,377233,378491,380072,381569,383174,384742,386362,387496,388978,390307,392048,393259,394503,395568,396493,397616,398688,399884,401043,402094,403020,404065,405184,406186,407004,407956,409121,410193,410903,411627,412221,413323,414065,414719,415275,415934,416996,417952,419066,420628,422121,423592,424553,425655,426387,427436,428607,429686,430187,430913,432122,433210,434178,435465,436728,437209,438252,439590,441147,442346,443787,445219,446608,448121,449748,451051,452457,453807,455116,456273,457158,457944,459131,460093,460978,462043,462943,463973,464816,465865,466971,468067,469028,470060,471162,472343,473263,474379,475636,476601,477619,478967,480012,480873,481949,482926,483836,484548,485567,486502,487624,488693,489836,490779,491658,492601,493719,494675,495712,496766,497761,498607,499645,500730,501901,502992,503820,504906,505969,507127,508266,509035,510055,511085,512126,513241,514172,514917,515692,516761,517987,519252,520491,521611],sizes:[1228,1032,1211,1161,987,978,1094,801,875,643,964,1140,1003,1130,1044,1149,1229,863,970,1092,1046,1053,1274,1152,1025,1335,975,1242,1301,1021,769,1052,1017,1392,1182,1271,1367,1248,1283,868,769,661,678,844,595,777,752,874,475,751,659,618,519,550,771,794,1160,864,784,636,819,532,274,607,564,794,577,632,616,585,713,547,658,873,656,509,531,628,725,851,912,586,662,792,783,766,628,816,791,537,660,745,783,1035,1098,1014,1006,986,914,756,662,608,965,530,764,640,943,675,765,627,525,705,887,745,866,909,1301,777,660,1044,530,557,859,971,711,511,1187,806,907,1049,877,584,647,711,574,277,1268,1150,379,329,858,1291,1175,1390,1242,1189,1262,1146,1069,1354,1198,1172,1011,1178,1461,1505,1523,1394,1140,1336,1448,1216,1103,889,1301,922,513,599,934,931,1073,752,966,674,1084,1089,1098,1326,1067,1241,1187,882,1294,1381,1053,1230,1278,1030,528,636,655,681,715,805,1008,866,1009,665,220,1032,1201,1120,1209,1259,744,898,788,1279,1304,727,999,975,1094,1173,1126,757,1238,899,714,1115,1070,1319,586,350,477,522,469,208,51,486,1198,572,459,495,562,727,974,1257,994,897,1125,1115,1241,1146,1250,1166,1362,831,575,1259,972,907,1426,766,337,927,825,945,718,950,783,1038,869,1156,841,930,613,452,648,478,535,931,291,1049,373,397,876,1212,1300,903,820,1435,1407,1429,1556,1339,1462,815,1114,1398,1594,1409,900,1198,1039,887,1270,1152,1046,1491,1313,1331,1267,865,830,845,985,1160,1475,1320,1051,1029,1044,1053,1059,1055,1051,1065,1063,1055,1063,1060,1063,1049,1062,1060,1053,1046,1062,1056,1052,1057,444,699,579,726,763,974,714,25,25,1007,1858,1781,1775,1973,1329,1467,1757,1184,1388,1724,1264,1616,1407,1133,1339,949,1076,1252,1417,1011,1269,1077,1379,1089,1031,1256,980,922,1247,1398,1047,1340,1008,1217,733,1470,949,1152,1345,1296,642,903,767,1182,1160,1153,1103,1406,1170,1258,1581,1497,1605,1568,1620,1134,1482,1329,1741,1211,1244,1065,925,1123,1072,1196,1159,1051,926,1045,1119,1002,818,952,1165,1072,710,724,594,1102,742,654,556,659,1062,956,1114,1562,1493,1471,961,1102,732,1049,1171,1079,501,726,1209,1088,968,1287,1263,481,1043,1338,1557,1199,1441,1432,1389,1513,1627,1303,1406,1350,1309,1157,885,786,1187,962,885,1065,900,1030,843,1049,1106,1096,961,1032,1102,1181,920,1116,1257,965,1018,1348,1045,861,1076,977,910,712,1019,935,1122,1069,1143,943,879,943,1118,956,1037,1054,995,846,1038,1085,1171,1091,828,1086,1063,1158,1139,769,1020,1030,1041,1115,931,745,775,1069,1226,1265,1239,1120,39],successes:[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]};compressedData.data=byteArray;assert(typeof Module.LZ4==="object","LZ4 not present - was your app build with  -s LZ4=1  ?");Module.LZ4.loadPackage({metadata:metadata,compressedData:compressedData});Module["removeRunDependency"]("datafile_regex.data")}Module["addRunDependency"]("datafile_regex.data");if(!Module.preloadResults)Module.preloadResults={};Module.preloadResults[PACKAGE_NAME]={fromCache:false};if(fetched){processPackageData(fetched);fetched=null}else{fetchedCallback=processPackageData}}if(Module["calledRun"]){runWithFS()}else{if(!Module["preRun"])Module["preRun"]=[];Module["preRun"].push(runWithFS)}};loadPackage({files:[{start:0,audio:0,end:46644,filename:"/lib/python3.8/site-packages/regex-2019.11.1-py3.8.egg-info"},{start:46644,audio:0,end:77742,filename:"/lib/python3.8/site-packages/regex/regex.py"},{start:77742,audio:0,end:288811,filename:"/lib/python3.8/site-packages/regex/test_regex.py"},{start:288811,audio:0,end:931148,filename:"/lib/python3.8/site-packages/regex/_regex.so"},{start:931148,audio:0,end:1073126,filename:"/lib/python3.8/site-packages/regex/_regex_core.py"},{start:1073126,audio:0,end:1073191,filename:"/lib/python3.8/site-packages/regex/__init__.py"}],remote_package_size:525746,package_uuid:"b93a2439-4400-45ea-b30e-7450a83612f2"})})();