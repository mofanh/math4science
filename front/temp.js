const emp = [];
emp[3.4] = "Casey Jones";
emp[1] = "Phil Lesh";
emp[2] = "August West";
emp.length = 10;

for (let a of emp) {
  console.log(a);
}
console.log("\n***\n");
for (let i = 0; i < emp.length; i++) {
  console.log(emp[i]);
}
console.log("\n***\n");
emp.map((element) => {
  console.log(element.slice(0, -1));
  return element.toUpperCase();
});

console.log(emp);
