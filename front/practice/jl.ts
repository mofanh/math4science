// 一段时间只允许调用一次函数
function jl(fn, wait) {
    let start  = new Date();
    return function() {
        let args = arguments;
        let end = new Date();
        if(end - start >= wait) {
            fn.apply(this, args);
            start = end;
        }
    }
}

const jl1 = jl(() => {
    console.log("1");
}, 2000)

jl1();

setTimeout(() => {
    jl1();
}, 1000)