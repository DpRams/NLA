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