(function () {
  var sheet = document.createElement("style");
  var select = document.createElement("select");
  select.innerHTML = '<option value=".cuda">CUDA</option> <option value=".pycuda">PyCUDA</option> <option value=".pyopencl">PyOpenCL</option> <option value=".cpp-openmp">C++ OpenMP</option> <option value=".cpp-openacc">C++ OpenACC</option> <option value=".f90-openmp">Fortan OpenMP</option> <option value=".f90-openacc">Fortran OpenACC</option>';
  document.head.appendChild(sheet);
  document.querySelector(".left-buttons").appendChild(select);

  function selectChange() {
    sheet.innerHTML = select.value + " { display: block; } " + select.value + "-code + pre { display: block; }";
    document.documentElement.style.setProperty("--language", '"' + select.options[select.selectedIndex].text + '"');
    window.localStorage.setItem("language", select.value);
  }

  select.onchange = selectChange;

  document.body.onload = function () {
    var language = window.localStorage.getItem("language");

    if (!language) {
      window.localStorage.setItem("language", ".cuda");
      select.querySelector('* [value=".cuda"]').selected = true;
    } else {
      select.querySelector('* [value="' + language + '"]').selected = true;
    }

    selectChange();
  }
})();