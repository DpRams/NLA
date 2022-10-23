function changingStatus(id){

  // changing status for deployment

  let deployRevoke_switch = document.getElementById(id)
  let model_id = document.getElementById("model_id")
  let deployStatus_id = document.getElementById("deployStatus_id")

  // console.log(deployRevoke_switch.dataset.modelid)
  // console.log(deployRevoke_switch.dataset.deploystatus)

  
  // false -(checked)> true 
  if (deployRevoke_switch.checked == true){

    let is_sure = window.confirm("是否要部署該模型?");
    if (is_sure == true){
      
      model_id.value = deployRevoke_switch.dataset.modelid
      deployStatus_id.value = statusTransforming(deployRevoke_switch.dataset.deploystatus)
      // console.log(model_id.value)
      // console.log(deployStatus_id.value)
      document.deployManagement.submit();

    }else {
      deployRevoke_switch.checked = false
    }
  
  // true -(unchecked)> false 
  }else if (deployRevoke_switch.checked == false){

    let is_sure = window.confirm("是否要撤除該模型?")
    if (is_sure == true){
      model_id.value = deployRevoke_switch.dataset.modelid
      deployStatus_id.value = statusTransforming(deployRevoke_switch.dataset.deploystatus)
      // console.log(model_id.value)
      // console.log(deployStatus_id.value)
      document.deployManagement.submit();

    }else {
      deployRevoke_switch.checked = true
    }
  }
}

// 假設初始狀態為 revoking, check 後，應該要變成 deploying 傳到後端
// ，但目前設計方法是會直接將原始的 status(revoking) 傳到後端(應該要是 deploying)
// 因此，此處將 狀態互換，再傳回後端，則後端收到的就是正確的操作動作(deploying)。
 
function statusTransforming(deploystatus){

    if (deploystatus == "revoking"){
      return "deploying"
    }else if (deploystatus == "deploying"){
      return "revoking"
    }

}