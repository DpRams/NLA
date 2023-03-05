function showLossFunctionEquation(){

    let select_loss = document.getElementById("select_loss");
    // let loss_equation = document.getElementById("lossEquation");
    let MSE_equation = document.getElementById("MSE");
    let RMSE_equation = document.getElementById("RMSE");
    let CROSSENTROPYLOSS_equation_b = document.getElementById("CROSSENTROPYLOSS_b");
    let CROSSENTROPYLOSS_equation_m = document.getElementById("CROSSENTROPYLOSS_m");

    // loss_equation.style.display = "none";
    MSE_equation.style.display = "none";
    RMSE_equation.style.display = "none";
    CROSSENTROPYLOSS_equation_b.style.display = "none";
    CROSSENTROPYLOSS_equation_m.style.display = "none";

    if (select_loss.value == "MSE"){
      // loss_equation.style.display = "block";
      MSE_equation.style.display = "block";
    }else if (select_loss.value == "RMSE"){
      // loss_equation.style.display = "block";
      RMSE_equation.style.display = "block";
    }else if (select_loss.value == "CROSSENTROPYLOSS"){
      // loss_equation.style.display = "block";
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

function AutoFillCrammingAndReorganizingRule(rule){

  let select_dataDescribing = document.getElementById("select_dataDescribing")
  // window.alert(select_dataDescribing.selectedIndex);


  if (rule == "cramming"){

    let select_crammingRule = document.getElementById("select_crammingRule")
    select_crammingRule.selectedIndex = select_dataDescribing.selectedIndex;
  
  }
  if (rule == "reorganizing"){

    let select_reorganizingRule = document.getElementById("select_reorganizingRule")
    select_reorganizingRule.selectedIndex = select_dataDescribing.selectedIndex;

  }
}

// function showModules(){

//   let cramming_switch = document.getElementById("cramming_switch")
//   let reorganizing_switch = document.getElementById("reorganizing_switch")

//   let cramming_mask = document.getElementById("cramming_mask")
//   let reorganizing_mask = document.getElementById("reorganizing_mask")

//   let select_crammingRule = document.getElementById("select_crammingRule")
//   let select_reorganizingRule = document.getElementById("select_reorganizingRule")

//   let reorganizingParamaters = document.getElementById("reorganizingParamaters")
  
//   if (cramming_switch.checked == true){
//     cramming_mask.style.color = "black";
//     cramming_mask.style.opacity = "1";
//     cramming_mask.style.pointerEvents = "";
//     AutoFillCrammingAndReorganizingRule('cramming');

//   }else if (cramming_switch.checked == false){
//     cramming_mask.style.color = "grey";
//     cramming_mask.style.opacity = "0.65";
//     cramming_mask.style.pointerEvents = "none";

//     // prevent that div has already disabled, but the select value is not "Disabled"
//     select_crammingRule.selectedIndex  = (select_crammingRule.options).length-1;
//   }
  
//   if (reorganizing_switch.checked == true){
//     reorganizing_mask.style.color = "black";
//     reorganizing_mask.style.opacity = "1";
//     reorganizing_mask.style.pointerEvents = "";
//     AutoFillCrammingAndReorganizingRule('reorganizing');

//     reorganizingParamaters.style.display = "";
    
//   }else if (reorganizing_switch.checked == false){
//     reorganizing_mask.style.color = "grey";
//     reorganizing_mask.style.opacity = "0.65";
//     reorganizing_mask.style.pointerEvents = "none";

//     // prevent that div has already disabled, but the select value is not "Disabled"
//     console.log((select_reorganizingRule.options).length-1);
//     select_reorganizingRule.selectedIndex  = (select_reorganizingRule.options).length-1;

//     reorganizingParamaters.style.display = "none";
//   }

// }


function showModules() {
  // Get all the necessary elements from the DOM
  const crammingSwitch = document.getElementById("cramming_switch");
  const reorganizingSwitch = document.getElementById("reorganizing_switch");
  const crammingMask = document.getElementById("cramming_mask");
  const reorganizingMask = document.getElementById("reorganizing_mask");
  const selectCrammingRule = document.getElementById("select_crammingRule");
  const selectReorganizingRule = document.getElementById("select_reorganizingRule");
  const reorganizingParameters = document.getElementById("reorganizingParamaters");
  
  // Check if the cramming switch is checked
  if (crammingSwitch.checked) {
    // If it is, set the appropriate styles and call the AutoFillCrammingAndReorganizingRule function with "cramming" as the parameter
    crammingMask.style.color = "black";
    crammingMask.style.opacity = "1";
    crammingMask.style.pointerEvents = "";
    AutoFillCrammingAndReorganizingRule('cramming');
  } else {
    // If it's not, set the appropriate styles and select the last option in the select element
    crammingMask.style.color = "grey";
    crammingMask.style.opacity = "0.65";
    crammingMask.style.pointerEvents = "none";
    selectCrammingRule.selectedIndex = selectCrammingRule.options.length - 1;
  }
  
  // Check if the reorganizing switch is checked
  if (reorganizingSwitch.checked) {
    // If it is, set the appropriate styles, call the AutoFillCrammingAndReorganizingRule function with "reorganizing" as the parameter, and show the reorganizing parameters element
    reorganizingMask.style.color = "black";
    reorganizingMask.style.opacity = "1";
    reorganizingMask.style.pointerEvents = "";
    AutoFillCrammingAndReorganizingRule('reorganizing');
    reorganizingParameters.style.display = "";
  } else {
    // If it's not, set the appropriate styles, select the last option in the select element, and hide the reorganizing parameters element
    reorganizingMask.style.color = "grey";
    reorganizingMask.style.opacity = "0.65";
    reorganizingMask.style.pointerEvents = "none";
    selectReorganizingRule.selectedIndex = selectReorganizingRule.options.length - 1;
    reorganizingParameters.style.display = "none";
  }
}

function commonLearningGoal(value){

  let learningGoal = document.getElementById("learningGoal");
  let regularizingLearningGoal = document.getElementById("regularizingLearningGoal");

  learningGoal.setAttribute("value", value);
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

