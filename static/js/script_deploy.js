function checkStatus(id){

  let deployRevoke_switch = document.getElementById(id)
  if (deployRevoke_switch.checked == false){
    return confirm("是否要撤除該模型?")
    
  }else {
    return confirm("是否要部署該模型?")
  }
}


function loadStatus(id){

  let deployRevoke_switch = document.getElementById(id);

  console.log(id);
  console.log(deployRevoke_switch.dataset.deploystatus);
  
  if (deployRevoke_switch.dataset.deploystatus == "revoking"){
    deployRevoke_switch.checked == False
  }
  else if (deployRevoke_switch.dataset.deploystatus == "revoking"){
    deployRevoke_switch.checked == True
  }


}