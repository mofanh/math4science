function myConcat(separator) {
    let result = "";
    for(let i = 1; i < arguments.length; i++)
    {
        result += arguments[i] + separator;
    }

    return result;
}

console.log(myConcat("、", "红", "橙", "蓝"))