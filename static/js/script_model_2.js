function showLossFunctionEquation(){

    let select_loss = document.getElementById("select_loss");
    let MSE_equation = document.getElementById("MSE");
    let RMSE_equation = document.getElementById("RMSE");
    let CROSSENTROPYLOSS_equation_b = document.getElementById("CROSSENTROPYLOSS_b");
    let CROSSENTROPYLOSS_equation_m = document.getElementById("CROSSENTROPYLOSS_m");

    MSE_equation.style.display = "none";
    RMSE_equation.style.display = "none";
    CROSSENTROPYLOSS_equation_b.style.display = "none";
    CROSSENTROPYLOSS_equation_m.style.display = "none";

    if (select_loss.value == "MSE"){
      MSE_equation.style.display = "block";
    }else if (select_loss.value == "RMSE"){
      RMSE_equation.style.display = "block";
    }else if (select_loss.value == "CROSSENTROPYLOSS"){
      CROSSENTROPYLOSS_equation_b.style.display = "block";
      CROSSENTROPYLOSS_equation_m.style.display = "block";
    }
}

function showInitializingParameters(){

  let select_initializingRule = document.getElementById("select_initializingRule");
  let regressionParameters = document.getElementById("regressionParameters");


  if (select_initializingRule.value == "LinearRegression"){
    regressionParameters.style.display = "block";
  }else{
    regressionParameters.style.display = "none";
  }

}

function showReorganzingingParameters(){

    let reorganizingRule = document.getElementById("select_reorganizingRule");
    let reorganizingParamaters = document.getElementById("reorganizingParamaters");

    reorganizingParamaters.style.display = "none";

    if (reorganizingRule.value != "Disabled"){
      reorganizingParamaters.style.display = "block";
    }
  }


function showMatchingParameters(){

  let select_matchingRule = document.getElementById("select_matchingRule");
  let matchingTimes = document.getElementById("p_matchingTimes");
  let matchingLearningGoal = document.getElementById("p_matchingLearningGoal");
  let matchingLearningRateLowerBound = document.getElementById("p_matchingLearningRateLowerBound");


  if (select_matchingRule.value == "EU"){

    matchingTimes.style.display = "block";
    matchingLearningGoal.style.display = "none";
    matchingLearningRateLowerBound.style.display = "none";

  }else if (select_matchingRule.value == "EU_LG"){

    matchingTimes.style.display = "block";
    matchingLearningGoal.style.display = "block";
    matchingLearningRateLowerBound.style.display = "none";

  }else if (select_matchingRule.value == "EU_LG_UA"){

    matchingTimes.style.display = "block";
    matchingLearningGoal.style.display = "block";
    matchingLearningRateLowerBound.style.display = "block";

  }
  else if (select_matchingRule.value == "Disabled"){

    matchingTimes.style.display = "none";
    matchingLearningGoal.style.display = "none";
    matchingLearningRateLowerBound.style.display = "none";

  }
  

}



function showModules(btnID){

  // after initializing_mask is enabled, button of cramming_mask, reorganizing_mask would be enabled
  let cramming_mask_e = document.getElementById("cramming_mask_e")
  let cramming_mask_d = document.getElementById("cramming_mask_d")
  let reorganizing_mask_e = document.getElementById("reorganizing_mask_e")
  let reorganizing_mask_d = document.getElementById("reorganizing_mask_d")

  let initializing_mask = document.getElementById("initializing_mask")
  let cramming_mask = document.getElementById("cramming_mask")
  let reorganizing_mask = document.getElementById("reorganizing_mask")

  let select_initializingRule = document.getElementById("select_initializingRule")
  let select_selectingRule = document.getElementById("select_selectingRule")
  let select_crammingRule = document.getElementById("select_crammingRule")
  let select_reorganizingRule = document.getElementById("select_reorganizingRule")


  if (btnID == "initializing_mask_e"){
  
    cramming_mask_e.disabled = false;
    cramming_mask_d.disabled = false;
    reorganizing_mask_e.disabled = false;
    reorganizing_mask_d.disabled = false;
  
  }else if (btnID == "initializing_mask_d"){

    cramming_mask_e.disabled = true;
    cramming_mask_d.disabled = true;
    reorganizing_mask_e.disabled = true;
    reorganizing_mask_d.disabled = true;
    
  }


  // show initializing_mask, cramming_mask, reorganizing_mask

  if (btnID == "initializing_mask_e"){

    initializing_mask.style.color = "black";
    initializing_mask.style.opacity = "1";
    initializing_mask.style.pointerEvents = "";

  }else if (btnID == "initializing_mask_d"){

    // prevent that button has been disabled, but the div has already enabled
    initializing_mask.style.color = "grey";
    initializing_mask.style.opacity = "0.65";
    initializing_mask.style.pointerEvents = "none";

    cramming_mask.style.color = "grey";
    cramming_mask.style.opacity = "0.65";
    cramming_mask.style.pointerEvents = "none";

    reorganizing_mask.style.color = "grey";
    reorganizing_mask.style.opacity = "0.65";
    reorganizing_mask.style.pointerEvents = "none";

    // prevent that div has already disabled, but the select value is not "Disabled"
    select_initializingRule.selectedIndex  = (select_initializingRule.options).length-1;
    select_selectingRule.selectedIndex  = (select_selectingRule.options).length-1;
    select_crammingRule.selectedIndex  = (select_crammingRule.options).length-1;
    select_reorganizingRule.selectedIndex  = (select_reorganizingRule.options).length-1;


  }else if (btnID == "cramming_mask_e"){
    cramming_mask.style.color = "black";
    cramming_mask.style.opacity = "1";
    cramming_mask.style.pointerEvents = "";
  }else if (btnID == "cramming_mask_d"){
    cramming_mask.style.color = "grey";
    cramming_mask.style.opacity = "0.65";
    cramming_mask.style.pointerEvents = "none";

    // prevent that div has already disabled, but the select value is not "Disabled"
    select_crammingRule.selectedIndex  = (select_crammingRule.options).length-1;
  }else if (btnID == "reorganizing_mask_e"){
    reorganizing_mask.style.color = "black";
    reorganizing_mask.style.opacity = "1";
    reorganizing_mask.style.pointerEvents = "";
  }else if (btnID == "reorganizing_mask_d"){
    reorganizing_mask.style.color = "grey";
    reorganizing_mask.style.opacity = "0.65";
    reorganizing_mask.style.pointerEvents = "none";

    // prevent that div has already disabled, but the select value is not "Disabled"
    select_reorganizingRule.selectedIndex  = (select_reorganizingRule.options).length-1;
  }






}

