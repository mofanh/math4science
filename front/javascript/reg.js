let text =
  "https://www.bing.com/search?q=js%E6%AD%A3%E5%88%99%E8%A1%A8%E8%BE%BE%E5%BC%8F&mkt=zh-CN https://kimi.moonshot.cn/chat/cobs0cqlnl9635ciafcg\
   http://47.113.106.215/weather/Anhui/Hefei%20Shi?id=c268c5e1494&lat=31.822809&lng=117.221803";

let text1 = "https://kimi.moonshot.cn/chat/cobs0cqlnl9635ciafcg";
let text2 =
  "http://47.113.106.215/weather/Anhui/Hefei%20Shi?id=c268c5e1494&lat=31.822809&lng=117.221803";

let reg = /((http(s)?):\/\/)[a-zA-Z0-9]*(\.[a-zA-Z0-9]*)*(\/[a-zA-Z0-9]*)*[a-zA-Z0-9%=?&.-]*/g;
// let result = reg.exec(text2);
let result = text.match(reg);
console.log(result);
