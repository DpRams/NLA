function changingStatus(id){

  // changing status for deployment

  let deployRevoke_switch = document.getElementById(id)
  let model_id = document.getElementById("model_id")

  // console.log(deployRevoke_switch.dataset.modelid)
  
  // false -(checked)> true 
  if (deployRevoke_switch.checked == true){

    let is_sure = window.confirm("是否要部署該模型?");
    if (is_sure == true){
      
      model_id.value = deployRevoke_switch.dataset.modelid
      // console.log(model_id.value)
      document.deployManagement.submit();

    }else {
      deployRevoke_switch.checked = false
    }
  
  // true -(unchecked)> false 
  }else if (deployRevoke_switch.checked == false){

    let is_sure = window.confirm("是否要撤除該模型?")
    if (is_sure == true){
      model_id.value = deployRevoke_switch.dataset.modelid
      // console.log(model_id.value)
      document.deployManagement.submit();

    }else {
      deployRevoke_switch.checked = true
    }
  }
}
