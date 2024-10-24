function fd(func, wait) {
  let timeout;

  return function() {
    let context = this;
    let args = arguments;
    clearTimeout(timeout)
    timeout = setTimeout(function(){
    func.apply(context, args);
    }, wait);
  } 
}

let fd1 = fd(function (){
  console.log('1')
}, 10000)
fd1()
fd1()