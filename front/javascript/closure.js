const createPet = function (name) {
    let sex;
  
    const pet = {
      // 在这个上下文中：setName(newName) 等价于 setName: function (newName)
      setName(newName) {
        name = newName;
      },
  
      getName() {
        return name;
      },
  
      getSex() {
        return sex;
      },
  
      setSex(newSex) {
        if (
          typeof newSex === "string" &&
          (newSex.toLowerCase() === "male" || newSex.toLowerCase() === "female")
        ) {
          sex = newSex;
        }
      },
    };
  
    return pet;
  };
  
  const pet = createPet("Vivie");
  console.log(pet.getName()); // Vivie
  
  pet.setName("Oliver");
  pet.setSex("male");
  console.log(pet.getSex()); // male
  console.log(pet.getName()); // Oliver
  