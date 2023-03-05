function restrictFileInputs() {
  // Get all the file input elements on the page

  const fileInputs = document.querySelectorAll('input[type="file"]');
  console.log(fileInputs);

  // Add a change event listener to each file input element
  fileInputs.forEach(fileInput => {
    fileInput.addEventListener('change', () => {
      // Disable all the other file input elements on the page that don't have a file selected
      fileInputs.forEach(input => {
        if (input !== fileInput && !input.files.length) {
          input.disabled = true;
        }
      });
    });

    // Enable all the file input elements when a file is removed from the current one
    fileInput.addEventListener('click', () => {
      fileInputs.forEach(input => {
        input.disabled = false;
      });
    });
  });
}

restrictFileInputs();