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

function showModules(){

  let initializing_switch = document.getElementById("initializing_switch")
  let cramming_switch = document.getElementById("cramming_switch")
  let reorganizing_switch = document.getElementById("reorganizing_switch")

  let initializing_mask = document.getElementById("initializing_mask")
  let cramming_mask = document.getElementById("cramming_mask")
  let reorganizing_mask = document.getElementById("reorganizing_mask")

  let select_initializingRule = document.getElementById("select_initializingRule")
  let select_selectingRule = document.getElementById("select_selectingRule")
  let select_crammingRule = document.getElementById("select_crammingRule")
  let select_reorganizingRule = document.getElementById("select_reorganizingRule")

  if (initializing_switch.checked == true){
  
    cramming_switch.disabled = false;
    reorganizing_switch.disabled = false;

  }else if (initializing_switch.checked == false){

    cramming_switch.disabled = true;
    cramming_switch.checked = false;
    reorganizing_switch.disabled = true;
    reorganizing_switch.checked = false;
    
  }

  if (initializing_switch.checked == true){

    initializing_mask.style.color = "black";
    initializing_mask.style.opacity = "1";
    initializing_mask.style.pointerEvents = "";
  }else if (initializing_switch.checked == false){
    
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

  }
  if (cramming_switch.checked == true){
    cramming_mask.style.color = "black";
    cramming_mask.style.opacity = "1";
    cramming_mask.style.pointerEvents = "";
  }else if (cramming_switch.checked == false){
    cramming_mask.style.color = "grey";
    cramming_mask.style.opacity = "0.65";
    cramming_mask.style.pointerEvents = "none";

    // prevent that div has already disabled, but the select value is not "Disabled"
    select_crammingRule.selectedIndex  = (select_crammingRule.options).length-1;
  }
  
  if (reorganizing_switch.checked == true){
    reorganizing_mask.style.color = "black";
    reorganizing_mask.style.opacity = "1";
    reorganizing_mask.style.pointerEvents = "";
  }else if (reorganizing_switch.checked == false){
    reorganizing_mask.style.color = "grey";
    reorganizing_mask.style.opacity = "0.65";
    reorganizing_mask.style.pointerEvents = "none";

    // prevent that div has already disabled, but the select value is not "Disabled"
    select_reorganizingRule.selectedIndex  = (select_reorganizingRule.options).length-1;
  }

}

function commonLearningGoal(value){

  let matchingLearningGoal = document.getElementById("matchingLearningGoal");
  let regularizingLearningGoal = document.getElementById("regularizingLearningGoal");

  matchingLearningGoal.setAttribute("value", value);
  regularizingLearningGoal.setAttribute("value", value);

  
}

function commonTuningTimes(value){

  let regularizingTimes = document.getElementById("regularizingTimes");

  regularizingTimes.setAttribute("value", value);

}

function commonLearningRateLowerBound(value){

  let regularizingLearningRateLowerBound = document.getElementById("regularizingLearningRateLowerBound");

  regularizingLearningRateLowerBound.setAttribute("value", value);

}

