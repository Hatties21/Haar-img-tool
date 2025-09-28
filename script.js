// ===== Shorthands & DOM =====
const $ = (s) => document.querySelector(s);

const fileInput = $("#file");
const btnReset = $("#btnReset");
const btnDownload = $("#btnDownload");
const btnToGray = $("#btnToGray");
const btnForward = $("#btnForward");
const btnInverse = $("#btnInverse");

const src = $("#src"),
  haar = $("#haar"),
  rec = $("#rec");
const infoSrc = $("#infoSrc"),
  infoHaar = $("#infoHaar"),
  infoRec = $("#infoRec");
const stats = $("#stats");

const orthonormEl = $("#orthonorm");

// Button groups
const levelsGroup = $("#levelsBtn"); // 1..5
const thrGroup = $("#thrBtn"); // 0,10,20,40,60
const procGroup = $("#procBtn"); // Auto,128,256,512,1024

// ===== State =====
let loadedImage = null; // HTMLImageElement (ảnh gốc đã nạp)
let originalImg = null; // ImageData (đang xử lý, grayscale)
let coeffs = null; // Float32Array hệ số Haar
let currentLevels = 1;

let levelsVal = 1; // giá trị mức biến đổi được chọn
let thrVal = 0; // ngưỡng epsilon
let procSizeVal = "Auto"; // 'Auto' | 128 | 256 | 512 | 1024

// ===== Helpers =====
function roundPow2(n) {
  let p = 1;
  while (p << 1 <= n && p < 1024) p <<= 1;
  return Math.max(8, p);
}

function toGrayscale(img) {
  const w = img.width,
    h = img.height;
  const out = new ImageData(w, h);
  const d = img.data,
    o = out.data;
  for (let i = 0; i < w * h; i++) {
    const r = d[i * 4],
      g = d[i * 4 + 1],
      b = d[i * 4 + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const v = Math.max(0, Math.min(255, Math.round(y)));
    o[i * 4] = o[i * 4 + 1] = o[i * 4 + 2] = v;
    o[i * 4 + 3] = 255;
  }
  return out;
}

function drawImageToCanvas(image, targetCanvas, size) {
  const tmp = document.createElement("canvas");
  const w = size || image.width;
  const h = size
    ? Math.round((image.height * size) / image.width)
    : image.height;
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(image, 0, 0, w, h);
  const sctx = targetCanvas.getContext("2d");
  targetCanvas.width = w;
  targetCanvas.height = h;
  sctx.drawImage(tmp, 0, 0);
  return sctx.getImageData(0, 0, w, h);
}

function putImageDataToCanvas(imgData, canvas) {
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  canvas.getContext("2d").putImageData(imgData, 0, 0);
}

function imgDataToArray(img) {
  const w = img.width,
    h = img.height;
  const arr = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) arr[i] = img.data[i * 4];
  return { arr, w, h };
}
function arrayToImgData(arr, w, h) {
  const out = new ImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    let v = Math.round(arr[i]);
    v = v < 0 ? 0 : v > 255 ? 255 : v;
    out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = v;
    out.data[i * 4 + 3] = 255;
  }
  return out;
}

// ===== Haar 1D / 2D =====
function haar1D(buffer, n, orthonorm = true) {
  const tmp = new Float32Array(n);
  const s = orthonorm ? Math.SQRT1_2 : 0.5;
  let k = 0;
  for (let i = 0; i < n; i += 2) {
    const a = buffer[i],
      b = buffer[i + 1];
    tmp[k] = (a + b) * s;
    tmp[k + n / 2] = (a - b) * s;
    k++;
  }
  buffer.set(tmp);
}
function ihaar1D(buffer, n, orthonorm = true) {
  const tmp = new Float32Array(n);
  let k = 0;
  for (let i = 0; i < n / 2; i++) {
    const s = buffer[i],
      d = buffer[i + n / 2];
    if (orthonorm) {
      tmp[k++] = (s + d) * Math.SQRT1_2;
      tmp[k++] = (s - d) * Math.SQRT1_2;
    } else {
      tmp[k++] = s + d;
      tmp[k++] = s - d;
    }
  }
  buffer.set(tmp);
}

function haar2D_forward(arr, w, h, L, orthonorm = true) {
  const data = new Float32Array(arr);
  let curW = w,
    curH = h;
  for (let level = 0; level < L; level++) {
    // rows
    for (let y = 0; y < curH; y++) {
      const row = new Float32Array(curW);
      for (let x = 0; x < curW; x++) row[x] = data[y * w + x];
      haar1D(row, curW, orthonorm);
      for (let x = 0; x < curW; x++) data[y * w + x] = row[x];
    }
    // cols
    for (let x = 0; x < curW; x++) {
      const col = new Float32Array(curH);
      for (let y = 0; y < curH; y++) col[y] = data[y * w + x];
      haar1D(col, curH, orthonorm);
      for (let y = 0; y < curH; y++) data[y * w + x] = col[y];
    }
    curW = Math.floor(curW / 2);
    curH = Math.floor(curH / 2);
  }
  return data;
}

function haar2D_inverse(data, w, h, L, orthonorm = true) {
  const arr = new Float32Array(data);
  let sizes = [],
    curW = w,
    curH = h;
  for (let i = 0; i < L; i++) {
    sizes.push([curW, curH]);
    curW = Math.floor(curW / 2);
    curH = Math.floor(curH / 2);
  }
  for (let level = L - 1; level >= 0; level--) {
    const [cw, ch] = sizes[level];
    // inverse cols
    for (let x = 0; x < cw; x++) {
      const col = new Float32Array(ch);
      for (let y = 0; y < ch; y++) col[y] = arr[y * w + x];
      ihaar1D(col, ch, orthonorm);
      for (let y = 0; y < ch; y++) arr[y * w + x] = col[y];
    }
    // inverse rows
    for (let y = 0; y < ch; y++) {
      const row = new Float32Array(cw);
      for (let x = 0; x < cw; x++) row[x] = arr[y * w + x];
      ihaar1D(row, cw, orthonorm);
      for (let x = 0; x < cw; x++) arr[y * w + x] = row[x];
    }
  }
  return arr;
}

function thresholdInPlace(data, thr) {
  if (!thr) return { zeroed: 0 };
  let zeroed = 0;
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i]) < thr) {
      data[i] = 0;
      zeroed++;
    }
  }
  return { zeroed };
}

function visualizeCoeffs(data, w, h) {
  let min = Infinity,
    max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const scale = max - min || 1;
  const img = new ImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = (data[i] - min) / scale;
    const g = Math.max(0, Math.min(255, Math.round(v * 255)));
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = g;
    img.data[i * 4 + 3] = 255;
  }
  return img;
}

function msePSNR(ref, test) {
  let mse = 0;
  for (let i = 0; i < ref.length; i++) {
    const e = ref[i] - test[i];
    mse += e * e;
  }
  mse /= ref.length;
  const psnr = mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
  return { mse, psnr };
}
function countNonzeros(data) {
  let c = 0;
  for (let i = 0; i < data.length; i++) if (data[i] !== 0) c++;
  return c;
}

// ===== Pipeline =====
async function loadImageFromFile(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = URL.createObjectURL(file);
  });
}

function prepareBase(img) {
  // chọn kích thước mục tiêu theo procSizeVal
  const target =
    procSizeVal === "Auto"
      ? roundPow2(Math.min(img.width, img.height))
      : parseInt(procSizeVal, 10);

  // vẽ ảnh -> src, rồi chuyển xám & hiển thị
  const srcData = drawImageToCanvas(img, src, target);
  originalImg = toGrayscale(srcData);
  putImageDataToCanvas(originalImg, src);
  infoSrc.textContent = `${originalImg.width}×${originalImg.height} px`;

  // bật các nút
  btnToGray.disabled = false;
  btnForward.disabled = false;
  btnReset.disabled = false;

  // reset kết quả cũ
  coeffs = null;
  haar.width = 0;
  rec.width = 0;
  stats.innerHTML = "";
  infoHaar.textContent = "";
  infoRec.textContent = "";
}

function doForward() {
  const { arr, w, h } = imgDataToArray(originalImg);
  currentLevels = levelsVal;
  coeffs = haar2D_forward(arr, w, h, currentLevels, orthonormEl.checked);
  thresholdInPlace(coeffs, thrVal);

  const vis = visualizeCoeffs(coeffs, w, h);
  putImageDataToCanvas(vis, haar);
  infoHaar.textContent = `${w}×${h} • L=${currentLevels} • ε=${thrVal} • NZ=${countNonzeros(
    coeffs
  )}`;

  btnInverse.disabled = false;
  btnDownload.disabled = true;

  const compression = (w * h) / Math.max(1, countNonzeros(coeffs));
  stats.innerHTML = `
    <div>Hệ số khác 0: <b>${countNonzeros(coeffs)}</b> / ${
    w * h
  } (tỷ lệ nén xấp xỉ: <b>${compression.toFixed(2)}:1</b>)</div>
    <div>Gợi ý: chọn ε lớn hơn để nén mạnh hơn; tăng L để có thêm băng con chi tiết.</div>
  `;
}

function doInverse() {
  if (!coeffs || !originalImg) return;
  const w = originalImg.width,
    h = originalImg.height;
  const recon = haar2D_inverse(
    coeffs,
    w,
    h,
    currentLevels,
    orthonormEl.checked
  );
  const recImg = arrayToImgData(recon, w, h);
  putImageDataToCanvas(recImg, rec);
  infoRec.textContent = `${w}×${h}`;

  const ref = imgDataToArray(originalImg).arr;
  const { mse, psnr } = msePSNR(ref, recon);
  stats.innerHTML += `<div style="margin-top:8px">Chất lượng: MSE=<b>${mse.toFixed(
    2
  )}</b>, PSNR=<b>${
    psnr === Infinity ? "∞" : psnr.toFixed(2) + " dB"
  }</b></div>`;

  btnDownload.disabled = false;
}

function resetAll() {
  coeffs = null;
  originalImg = null;
  loadedImage = null;
  src.width = haar.width = rec.width = 0;
  src.height = haar.height = rec.height = 0;
  infoSrc.textContent = infoHaar.textContent = infoRec.textContent = "";
  stats.innerHTML = "";
  btnToGray.disabled = true;
  btnForward.disabled = true;
  btnInverse.disabled = true;
  btnDownload.disabled = true;
  btnReset.disabled = true;
}

function downloadResult() {
  const link = document.createElement("a");
  link.download = "haar_result.png";
  link.href = rec.toDataURL("image/png");
  link.click();
}

// ===== Events =====
fileInput.addEventListener("change", async (e) => {
  if (!e.target.files?.length) return;
  resetAll();
  loadedImage = await loadImageFromFile(e.target.files[0]);
  prepareBase(loadedImage);
});

btnReset.addEventListener("click", resetAll);

btnToGray.addEventListener("click", () => {
  if (!src.width) return;
  const ctx = src.getContext("2d");
  const d = ctx.getImageData(0, 0, src.width, src.height);
  originalImg = toGrayscale(d);
  putImageDataToCanvas(originalImg, src);
});

btnForward.addEventListener("click", doForward);
btnInverse.addEventListener("click", doInverse);
btnDownload.addEventListener("click", downloadResult);

// Toggle chuẩn hoá
orthonormEl.addEventListener("change", () => {
  if (originalImg) {
    // đổi chuẩn -> cần tính lại nếu đang có coeffs
    coeffs = null;
    haar.width = 0;
    rec.width = 0;
    stats.innerHTML = "";
    infoHaar.textContent = "";
    infoRec.textContent = "";
  }
});

// ===== Button groups =====
// Levels
levelsGroup.querySelectorAll("button").forEach((b) => {
  b.addEventListener("click", () => {
    levelsGroup
      .querySelectorAll("button")
      .forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    levelsVal = parseInt(b.dataset.level, 10);
    if (coeffs) doForward();
  });
});

// Threshold
thrGroup.querySelectorAll("button").forEach((b) => {
  b.addEventListener("click", () => {
    thrGroup
      .querySelectorAll("button")
      .forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    thrVal = parseInt(b.dataset.thr, 10);
    if (coeffs) doForward();
  });
});

// Processing size (áp dụng ngay nếu đã có ảnh)
procGroup.querySelectorAll("button").forEach((b) => {
  b.addEventListener("click", () => {
    procGroup
      .querySelectorAll("button")
      .forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    procSizeVal = b.dataset.size; // 'Auto' or number as string
    if (loadedImage) {
      // re-prepare từ ảnh gốc đã nạp với kích thước mới
      prepareBase(loadedImage);
    }
  });
});

// Hover preview (giữ Shift để xem ảnh gốc trên canvas tái tạo)
rec.addEventListener("mousemove", (e) => {
  if (!e.shiftKey || !originalImg) return;
  rec.getContext("2d").putImageData(originalImg, 0, 0);
});
rec.addEventListener("mouseleave", () => {
  if (!coeffs || !originalImg) return;
  doInverse();
});
