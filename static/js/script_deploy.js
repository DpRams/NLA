function checkStatus(){

  let deployRevoke_switch = document.getElementById("deployRevoke_switch")
  if (deployRevoke_switch.checked == false){
    return confirm("是否要撤除該模型?")
    
  }else {
    return confirm("是否要部署該模型?")
  }
}
