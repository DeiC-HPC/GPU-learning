(function () {
  var sheet = document.createElement("style");
  var select = document.createElement("select");
  select.innerHTML = '<option value=".openmp">openmp</option> <option value=".openacc">openacc</option> <option value=".cuda">CUDA</option>';

  function selectChange() {
    sheet.innerHTML = select.value + " { display: initial; };";
    window.localStorage.setItem("language", select.value);
  }

  select.onchange = selectChange;

  document.body.onload = function () {
    document.head.appendChild(sheet);
    document.querySelector(".left-buttons").appendChild(select);
    var language = window.localStorage.getItem("language");

    if (!language) {
      window.localStorage.setItem("language", ".cuda");
      select.querySelector('* [value=".cuda"]').selected = true;
    }

    else {
      select.querySelector('* [value="' + language + '"]').selected = true;
    }

    selectChange();
  }
})()
