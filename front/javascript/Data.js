var today = new Date();
var endYear = new Date(1995, 11, 31, 23, 59, 59, 999); // 设置日和月，注意，月份是 0-11
endYear.setFullYear(today.getFullYear()); // 把年设置为今年
var msPerDay = 24 * 60 * 60 * 1000; // 每天的毫秒数
var daysLeft = (endYear.getTime() - today.getTime()) / msPerDay;
var daysLeft = Math.round(daysLeft); //返回今年剩下的天数
console.log(daysLeft)
