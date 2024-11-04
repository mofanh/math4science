Function.prototype._bind = function (ctx){
    const args = ctx;
    args.fn = this;
    return function() {
        args.fn()
    }
}

function aaa(){
    console.log('aaa--', this.a)
}

const tmp = {
    a: 'sdfdf'
}

const bindFun = aaa._bind(tmp)

bindFun()