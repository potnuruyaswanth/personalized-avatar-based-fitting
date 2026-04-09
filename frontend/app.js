const personInput = document.getElementById("person");
const clothInput = document.getElementById("cloth");
const personPreview = document.getElementById("person-preview");
const clothPreview = document.getElementById("cloth-preview");
const resultImage = document.getElementById("result-image");
const statusText = document.getElementById("status");
const tryOnButton = document.getElementById("tryon-btn");
const downloadLink = document.getElementById("download-link");

function showPreview(input, imageElement) {
  const file = input.files?.[0];
  if (!file) {
    imageElement.removeAttribute("src");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    imageElement.src = reader.result;
  };
  reader.readAsDataURL(file);
}

personInput.addEventListener("change", () => showPreview(personInput, personPreview));
clothInput.addEventListener("change", () => showPreview(clothInput, clothPreview));

tryOnButton.addEventListener("click", async () => {
  const person = personInput.files?.[0];
  const cloth = clothInput.files?.[0];

  if (!person || !cloth) {
    statusText.textContent = "Please upload both person and cloth images.";
    return;
  }

  const data = new FormData();
  data.append("person", person);
  data.append("cloth", cloth);

  try {
    tryOnButton.disabled = true;
    statusText.textContent = "Generating your virtual try-on...";

    const response = await fetch("/api/try-on", {
      method: "POST",
      body: data,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "Request failed");
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);

    resultImage.src = objectUrl;
    downloadLink.href = objectUrl;
    statusText.textContent = "Done. You can download the result.";
  } catch (error) {
    statusText.textContent = `Error: ${error.message}`;
  } finally {
    tryOnButton.disabled = false;
  }
});
