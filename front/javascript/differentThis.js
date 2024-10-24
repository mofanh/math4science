// 每个函数都有自己的this，这使得函数内嵌套函数也有自己的this
function Person() {
  // 有的人习惯用 `that` 而不是 `self`。
  // 请选择一种方式，并保持前后代码的一致性
  const self = this;
  self.age = 0;

  setInterval(function growUp() {
    // 回调引用 `self` 变量，其值为预期的对象。
    self.age++;
    console.log(self.age);
  }, 1000);
}

// 为了解决函数内函数也有自己的this，引入了箭头函数（没有自己的this）
function Person() {
  this.age = 0;

  setInterval(() => {
    this.age++; // 这里的 `this` 正确地指向 person 对象
    console.log(this.age);
  }, 1000);
}

const p = new Person();
