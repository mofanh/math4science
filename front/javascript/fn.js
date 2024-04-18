function map(fn, data) {
  let out = new Array(data.length);

  for (let i = 0; i < out.length; i++) {
    out[i] = fn(data[i]);
  }

  return out;
}

const cube = (num) => {
  return num * num * num;
};

let data = [1, 2, 3, 4];

console.log(map(cube, data));
