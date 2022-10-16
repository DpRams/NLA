function createOrSelect(){

  let select_dataUse = document.getElementById("select_dataUse")
  let selectDirectory = document.getElementById("selectDirectory")
  let createDirectory = document.getElementById("createDirectory")

  selectDirectory.style.display = "none";
  createDirectory.style.display = "none";

  if (select_dataUse.selectedIndex == 0){
    createDirectory.style.display = "block";

  }else if (select_dataUse.selectedIndex == 1){
    selectDirectory.style.display = "block";

  }
}
