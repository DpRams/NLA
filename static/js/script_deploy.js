function changingStatus(id){

  // changing status for deployment

  let deployRevoke_switch = document.getElementById(id)
  let modelPklFile = document.getElementById("modelPklFile-id")
  // console.log(deployRevoke_switch.dataset.modelname)
  
  // false -(checked)> true 
  if (deployRevoke_switch.checked == true){

    let is_sure = window.confirm("是否要部署該模型?");
    if (is_sure == true){
      
      modelPklFile.value = deployRevoke_switch.dataset.modelname
      // console.log(modelPklFile.value)
      document.deployManagement.submit();

    }else {
      deployRevoke_switch.checked = false
    }
  
  // true -(unchecked)> false 
  }else if (deployRevoke_switch.checked == false){

    let is_sure = window.confirm("是否要撤除該模型?")
    if (is_sure == true){
      modelPklFile.value = deployRevoke_switch.dataset.modelname
      // console.log(modelPklFile.value)
      document.deployManagement.submit();

    }else {
      deployRevoke_switch.checked = true
    }
  }
}

// 這個部分用 Jinja 的 if 寫掉了
// function loadingStatus(id){

//   // loading the deployStatus from record(csv)

//   let deployRevoke_switch = document.getElementById(id);

//   console.log(id);
//   console.log(deployRevoke_switch.dataset.deploystatus);

//   if (deployRevoke_switch.dataset.deploystatus == "revoking"){
//     deployRevoke_switch.checked == False
//   }
//   else if (deployRevoke_switch.dataset.deploystatus == "deploying"){
//     deployRevoke_switch.checked == True
//   }
// }