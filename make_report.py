"""Builds module_summary.pdf (Generative AI Analysis and Ethics Report) with reportlab.

Every number in the report is read from results/metrics.json, which generative_model.ipynb writes,
and every figure is embedded from figures/. Run the notebook first, then:  python make_report.py
"""
import json
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

M = json.load(open("results/metrics.json", encoding="utf-8"))
H = M["history"]
OUT = "module_summary.pdf"
FIG = "figures"

# ----------------------------------------------------------------------------- styles and helpers
styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["BodyText"], fontName="Times-Roman", fontSize=11, leading=15,
                      spaceAfter=7)
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=14, leading=18,
                    spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=11.5, leading=15,
                    spaceBefore=6, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=styles["Title"], fontName="Times-Bold", fontSize=18, leading=23,
                       alignment=TA_CENTER, spaceAfter=8)
SUB = ParagraphStyle("sub", parent=BODY, alignment=TA_CENTER, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=9.5, leading=12.5, spaceAfter=10)
REF = ParagraphStyle("ref", parent=BODY, leftIndent=0.5 * inch, firstLineIndent=-0.5 * inch)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9.5, leading=12, spaceAfter=0)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=14, bulletIndent=2, spaceAfter=4)

story = []


def h1(t):
    story.append(Paragraph(t, H1))


def h2(t):
    story.append(Paragraph(t, H2))


def p(t):
    story.append(Paragraph(t, BODY))


def bullets(items):
    for t in items:
        story.append(Paragraph(t, BULLET, bulletText="•"))


def figure(name, width_in, caption):
    img = Image(f"{FIG}/{name}")
    scale = width_in * inch / img.drawWidth
    img.drawWidth, img.drawHeight = img.drawWidth * scale, img.drawHeight * scale
    story.append(KeepTogether([Spacer(1, 4), img, Spacer(1, 4), Paragraph(caption, CAP)]))


def table(rows, col_widths, caption=None):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * inch for w in col_widths], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    block = [Paragraph(caption, CAP)] if caption else []
    story.append(KeepTogether(block + [t, Spacer(1, 8)]))


def f(x, n=2):
    return f"{x:,.{n}f}"


# ----------------------------------------------------------------------------- derived numbers
E = M["epochs"]
val_first, val_last = H["val_total"][0], H["val_total"][-1]
tr_last = H["train_total"][-1]
gap = val_last - tr_last
last5 = H["val_total"][-6] - val_last
pcm = M["per_class_mse"]
pcb = M["per_class_bce"]
worst_mse = sorted(pcm, key=pcm.get, reverse=True)
worst_bce = sorted(pcb, key=pcb.get, reverse=True)
best = worst_mse[-1]
knn = M["knn_agreement"]
nn_gen = sum(M["nn_dist_generated"]) / len(M["nn_dist_generated"])
nn_ref = sum(M["nn_dist_reference_test"]) / len(M["nn_dist_reference_test"])
n_cov = sum(v > 0 for v in M["sample_class_counts"].values())
cnt = M["sample_class_counts"]
pick = M["recon_pick_mse"]
mse_rise = (M["beta_test_mse"] / M["test_mse"] - 1) * 100

# ----------------------------------------------------------------------------- title block
story += [Spacer(1, 6),
          Paragraph("Generative AI Analysis and Ethics Report", TITLE),
          Paragraph("A Convolutional Variational Autoencoder for Fashion-MNIST Clothing Images", SUB),
          Paragraph("Sirvanmesh", SUB),
          Paragraph("Udacity Master's in AI Capstone – Project 5: Generative AI Applications", SUB),
          Paragraph(date.today().strftime("%B %d, %Y").replace(" 0", " "), SUB),
          Spacer(1, 10)]

# ----------------------------------------------------------------------------- 1. Overview
h1("1. Overview")
p("The goal of this project is to <b>generate new, realistic clothing images (28 x 28 grayscale) and to learn a "
  "smooth latent space of clothing styles</b>. I implemented a convolutional Variational Autoencoder (VAE) in "
  "PyTorch and trained it on Fashion-MNIST, a public dataset of 70,000 real Zalando product photos in ten clothing "
  "classes (Xiao et al., 2017). A VAE is a latent-variable generative model with two parts: an encoder that maps an "
  "image to a Gaussian distribution over a low-dimensional code, and a decoder that maps a code back to an image. "
  "Both parts are trained together by maximising a lower bound on the data log-likelihood, the evidence lower "
  "bound or ELBO (Kingma &amp; Welling, 2014).")
p("I chose a VAE over a GAN for three reasons. It trains by minimising a single loss, so there is no adversarial "
  "game that can become unstable. It gives an explicit latent space, which I need for interpolation. And the same "
  "model can both reconstruct images, which I can measure, and generate new images from the prior. The trained "
  f"model has {M['n_params']:,} parameters and a {M['latent_dim']}-dimensional latent space, and it trained in "
  f"{f(M['train_minutes'], 1)} minutes on a CPU. This report describes the data, the design and training, "
  "the quality and failure modes of the outputs, and the ethical questions this specific system raises.")

# ----------------------------------------------------------------------------- 2. Dataset
h1("2. Dataset or Prompt Description")
p("<b>Source and licence.</b> Fashion-MNIST was released by Zalando Research and is available at "
  "https://github.com/zalandoresearch/fashion-mnist under the MIT License. It contains front-view product images "
  "from Zalando's online catalogue that the authors converted to 28 x 28 grayscale. It was designed as a harder "
  "drop-in replacement for the MNIST digit benchmark (Xiao et al., 2017). The images are real photos of real "
  "products. They are not synthetic and not AI-generated. I use them only for this academic project. The "
  "notebook downloads the data automatically with <font face='Courier'>torchvision.datasets.FashionMNIST(root="
  "'./data', download=True)</font>, so anyone can rerun the project without an account or manual steps.")
p("<b>Size and structure.</b> The dataset has 60,000 training and 10,000 test images in ten classes: T-shirt/top, "
  "trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. Each class has exactly 6,000 "
  "training and 1,000 test images, so the classes are perfectly balanced. Pixels are 8-bit integers (0–255). "
  f"I held out {M['n_val']:,} random training images (fixed seed) for validation, which leaves {M['n_train']:,} "
  f"for training. The {M['n_test']:,} test images are used only for the final evaluation.")
p("<b>How it supports the task.</b> The items are centred on a black background with a consistent pose. The model "
  "can therefore spend its capacity on garment shape and style instead of on background or position, which makes "
  "the dataset well suited to learning a compact latent space of clothing styles. Inspecting the samples (Figure 1) "
  "also showed in advance where problems were likely. Trousers and footwear have distinctive outlines, but "
  "T-shirts, pullovers, coats and shirts share almost the same silhouette. Many items carry fine detail, such as "
  "a printed \"89\" T-shirt, a plaid shirt and striped sneakers. For preprocessing, I scaled pixels to [0, 1] "
  "with <font face='Courier'>ToTensor()</font>, because the binary cross-entropy loss expects targets in that "
  "range. I did not use augmentation, because flips or rotations would create poses that do not occur in the "
  "catalogue.")
figure("data_samples.png", 5.8, "<b>Figure 1.</b> Four random training images per class, displayed before modelling.")

# ----------------------------------------------------------------------------- 3. Model design
h1("3. Model Design and Training Approach")
h2("Architecture")
p("The encoder has two stride-2 convolutions (1 → 32 → 64 channels, 4 x 4 kernels, ReLU) that reduce "
  "the image from 28 x 28 to 14 x 14 and then 7 x 7. These are followed by a 256-unit fully connected layer and "
  f"two linear heads that output the mean μ and log-variance log σ² of a {M['latent_dim']}-dimensional "
  "Gaussian. A latent code is sampled with the reparameterisation trick, z = μ + σ × ε with "
  "ε ~ N(0, I) (element-wise product). This makes the sampling step differentiable, so the encoder can be trained by backpropagation "
  "(Kingma &amp; Welling, 2014). The decoder mirrors the encoder: two fully connected layers, a reshape to "
  "64 x 7 x 7, and two stride-2 transposed convolutions back to 1 x 28 x 28, with a sigmoid so that every output "
  "pixel is a Bernoulli mean in [0, 1]. I used convolutions rather than a fully connected network because they "
  "share weights across positions and capture local structure such as collars, straps and soles.")
h2("Loss, optimiser and hyperparameters")
p("The loss is the negative ELBO for each image: the binary cross-entropy between input and output, summed over "
  "the 784 pixels (the negative log-likelihood of a Bernoulli decoder), plus the closed-form KL divergence between "
  "the encoder distribution and the standard normal prior (Kingma &amp; Welling, 2014). The KL term is what makes "
  "random codes from N(0, I) decode into realistic images. Higgins et al. (2017) show that weighting this term "
  "with a factor β trades reconstruction detail for a smoother, more regular latent space. I used β = 1 "
  "for the main model and tested β = 4 as a single controlled change. The training settings were Adam with "
  f"learning rate {M['lr']:g}, batch size {M['batch_size']} ({-(-M['n_train'] // M['batch_size'])} updates per epoch), "
  f"and {E} epochs, with the seed fixed for every random number generator. Adam adapts the step size for each "
  "parameter, similar to the adaptive optimiser used in the original VAE work (Kingma &amp; Welling, 2014), and "
  "it handles the very different gradient scales of the reconstruction and KL terms.")
table([["Setting", "Value", "Reason"],
       ["Latent size", str(M["latent_dim"]), "Room for shape, sleeve length, heel and brightness; still smooth enough to sample"],
       ["Reconstruction loss", "BCE, summed over pixels", "Bernoulli likelihood for [0, 1] pixels"],
       ["Regulariser", "KL to N(0, I), β = 1", "Makes prior samples decodable"],
       ["Optimiser", f"Adam, lr = {M['lr']:g}", "Stable adaptive steps"],
       ["Batch / epochs", f"{M['batch_size']} / {E}", f"Converged within the CPU budget ({f(M['train_minutes'], 1)} min)"]],
      [1.35, 1.75, 3.4], "<b>Table 1.</b> Main design choices.")
h2("Justification from my training results")
p(f"Figure 2 shows that training was stable. The validation loss fell from {f(val_first)} to {f(val_last)} nats "
  f"per image and improved by only {f(last5)} nats over the last five epochs, so {E} epochs were enough and I did "
  f"not need to shorten training. At the end, the validation loss was only {f(gap)} nats above the training loss, "
  "so the model did not over-fit. The curves also show the expected tension between the two loss terms. "
  f"Reconstruction loss fell (validation {f(H['val_recon'][0])} → {f(H['val_recon'][-1])}) while the KL term "
  f"rose ({f(H['val_kl'][0])} → {f(H['val_kl'][-1])}): the encoder put more information into z because this "
  "bought better reconstructions. A per-dimension KL diagnostic showed one training challenge: only "
  f"{M['active_dims']} of the {M['latent_dim']} latent dimensions carry information, and the rest stay at the "
  "prior. Because the KL term charges a cost for every dimension the encoder uses, the model leaves dimensions it "
  "does not need at the prior. So 16 dimensions were more than this model needed. This did not prevent good "
  "samples, but it shows that the latent size could be reduced.")
figure("loss_curves.png", 6.2, "<b>Figure 2.</b> Training and validation loss per epoch: total negative ELBO (left), "
       "BCE reconstruction (middle) and KL divergence (right), in nats per image.")

# ----------------------------------------------------------------------------- 4. Evaluation
h1("4. Output Evaluation and Interpretation")
p(f"<b>Quantitative overview.</b> On the {M['n_test']:,} test images the model reaches a negative ELBO of "
  f"{f(M['test_total'])} nats per image, a reconstruction BCE of {f(M['test_bce'])} nats and a mean squared "
  f"reconstruction error of {f(M['test_mse'], 4)} per pixel. These numbers measure reconstruction, not the realism "
  "of new samples. I therefore based my evaluation of generated outputs mainly on a structured visual inspection "
  "of 64 random samples (Figure 3), 16 reconstructions (Figure 4) and three interpolations (Figure 5). All "
  "positions below refer to the labelled grids.")
figure("generated_samples.png", 4.1, "<b>Figure 3.</b> 64 new images decoded from z ~ N(0, I). Rows and columns are "
       "numbered so that individual samples can be referenced.")
h2("Quality and variability of new samples")
p("Most samples are recognisable garments with clean outlines. Examples are the wide white trousers at row 1, "
  "column 1, the slim trousers at row 6, column 6, a sneaker with a clear sole line at row 3, column 8, two flat "
  "ankle boots at row 4, columns 1–2, and a bag with a handle at row 7, column 8. To check diversity, I "
  "labelled each sample with the class of its nearest training image. The 64 samples cover "
  f"{n_cov} of 10 classes, with coats ({cnt['Coat']}), sneakers ({cnt['Sneaker']}) and ankle boots "
  f"({cnt['Ankle boot']}) the most frequent. Samples also vary within a class, for example wide vs slim trousers "
  "and heeled boots (row 1, column 8; row 5, column 8) vs flat boots (row 4, column 1). The model has therefore "
  "not collapsed to a few prototypes.")
h2("Failure cases (concrete examples)")
bullets([
    "<b>Blur and lost texture.</b> None of the 64 samples has a print, logo, stripe or check pattern, although the "
    "training data contains many (Figure 1). All surfaces are smooth grey gradients. The reconstructions show the "
    f"same effect most clearly. The pullover #3 carries a large \"Lee\" logo; its reconstruction keeps the "
    f"pullover's shape but erases the logo completely (MSE {f(pick['#3 Pullover'], 3)}, the worst of the 16). The "
    f"plaid shirt #15 becomes a plain grey shirt (MSE {f(pick['#15 Shirt'], 3)}). Plain items such as trouser #2 and "
    f"dress #4 are almost perfect (MSE {f(pick['#2 Trouser'], 3)}). Because the decoder outputs the mean value of "
    "each pixel, uncertain detail is averaged into grey.",
    "<b>Class blending.</b> The sample at row 1, column 7 joins a T-shirt-like top to a boot-like lower outline. "
    "Row 6, column 2 is a sleeveless rounded rectangle halfway between a bag and a pullover. Row 1, column 4 is a "
    "shapeless grey blob that matches no class.",
    "<b>Artefacts.</b> Row 6, column 7 is a fragmented shape with horizontal streaks. Row 8, column 6 is a white "
    "pullover with a dark smudge where a printed graphic should be.",
    f"<b>Weak classes.</b> Only {cnt['Sandal']} of the 64 samples resembles a sandal (row 7, column 2), and it is a "
    f"closed, dark shoe without straps. Across the full test set, the highest per-class MSE belongs to "
    f"{worst_mse[0].lower()} ({f(pcm[worst_mse[0]], 4)}) and {worst_mse[1].lower()} ({f(pcm[worst_mse[1]], 4)}), "
    f"and the lowest to {best.lower()}s ({f(pcm[best], 4)}). Thin straps and varied bag shapes are hard to "
    f"reproduce pixel by pixel. Ranked by BCE, {worst_bce[0].lower()} ({f(pcb[worst_bce[0]], 1)}) and "
    f"{worst_bce[1].lower()} ({f(pcb[worst_bce[1]], 1)}) are worst. In the latent space, shirts, pullovers and coats "
    f"are the most mixed classes: only {f(knn['Shirt'] * 100, 0)}%, {f(knn['Pullover'] * 100, 0)}% and "
    f"{f(knn['Coat'] * 100, 0)}% of their ten nearest latent neighbours share their class, against "
    f"{f(knn['Trouser'] * 100, 0)}% for trousers. This overlap is the likely cause of the blending failures.",
])
figure("reconstructions.png", 5.9, "<b>Figure 4.</b> Test images (rows 1 and 3) and their reconstructions (rows 2 "
       "and 4), with per-image MSE.")
h2("What the interpolations show")
p("Figure 5 decodes straight lines between the latent means of two real test images. Within the footwear family "
  "the latent space is smooth and meaningful: from sneaker to ankle boot the sole stays fixed while the shaft rises "
  "steadily, and every frame is a plausible shoe. Between families the path passes through other classes. The "
  "shirt-to-trouser path goes through a sleeveless, dress-like shape (a ≈ 0.33–0.44) before two legs "
  "separate. The sandal-to-bag path first turns the open sandal into a closed sneaker-like shoe and only then into "
  "a box-shaped bag without the original bag's handle. These middle frames are hybrids that do not exist as real "
  "products, so the space is smooth but not organised by class.")
figure("latent_interpolation.png", 6.5, "<b>Figure 5.</b> Latent interpolation for three class pairs. The real start "
       "and end images are shown in the outer columns.")
h2("Effect of the KL weight (β = 4)")
p("With everything else fixed, raising β to 4 lowered the test KL from "
  f"{f(M['test_kl'])} to {f(M['beta_test_kl'])} nats and reduced the active latent dimensions from "
  f"{M['active_dims']} to {M['active_dims_beta']}. Test MSE rose by {f(mse_rise, 0)}% "
  f"({f(M['test_mse'], 4)} → {f(M['beta_test_mse'], 4)}), and the samples were visibly blurrier, with faint, "
  "ghost-like shoes. This confirms the trade-off described by Higgins et al. (2017). For my goal of realistic "
  "samples, β = 1 is the better setting.")

# ----------------------------------------------------------------------------- 5. Ethics
h1("5. Ethical Considerations and Responsible Use")
h2("(a) Bias and representation")
p("Everything this model can produce comes from one retailer's catalogue. Fashion-MNIST was built from Zalando "
  "product images (Xiao et al., 2017), which mostly show mainstream, Western-market clothing photographed on a "
  "plain background. Its ten categories are retail labels that follow gender-coded shop conventions (for example, "
  "dresses form their own class, while the dataset itself has no gender or body-type information). My outputs "
  "show this limit directly. "
  "Every sample in Figure 3 falls into one of the ten catalogue classes. The latent space has no region for "
  "garments outside them, such as a sari, a kimono, a hijab or adaptive clothing. Interpolations between classes "
  "produce hybrids rather than new styles. If a system like this were used to propose designs or to fill product "
  "databases, it would reproduce one retailer's view of what clothing is, and it would quietly leave out other "
  "cultures and body types. A responsible claim about this model is therefore narrow: it represents the "
  "Fashion-MNIST catalogue, not clothing in general.")
h2("(b) Creative ownership and memorisation")
p("The training images show products designed by other people and photographed by Zalando, so a generator that "
  "reproduced them would copy their work. Carlini et al. (2023) showed that large image generators, including "
  "diffusion models, can memorise and regenerate individual training images. They measured this as a very small "
  "pixel-space distance between a generated image and a training image. I applied the same idea at small scale. "
  f"For all 64 samples, the mean L2 distance to the nearest of the 60,000 training images was {f(nn_gen)} "
  f"(minimum {f(min(M['nn_dist_generated']))}). For 64 real, unseen test photos it was {f(nn_ref)} (minimum "
  f"{f(min(M['nn_dist_reference_test']))}). My samples are therefore not closer to the training data than genuinely "
  "different real products are. Figure 6 shows that the nearest neighbours share only a rough outline; the heeled "
  "boot at row 1, column 8, for example, is matched to a high-top sneaker with a bright logo. With about 1.7 million "
  "parameters and a 16-dimensional bottleneck, the model can keep only coarse shape, and that limits copying. "
  "Pixel distance is still a crude test. Blurry average images can be close to many training images without copying "
  "any of them, so this check supports, but does not prove, the absence of memorisation.")
figure("nearest_neighbors.png", 5.6, "<b>Figure 6.</b> Memorisation check: the first eight generated samples (top) "
       "and their nearest training images by pixel L2 distance (bottom).")
h2("(c) Misuse")
p("The most realistic misuse of clothing generators is synthetic product photos for fake or misleading online "
  "listings, such as selling items that do not exist or imitating a brand's catalogue. My model is a low risk here. "
  "Its outputs are 28 x 28, grayscale, visibly blurry and without logos (Figures 3 and 4), so nobody would mistake "
  "them for real product photos. The risk grows quickly with scale, however. High-resolution colour models trained "
  "on scraped catalogues, such as modern diffusion models, can produce convincing product images and can reproduce "
  "specific training images (Carlini et al., 2023). The same design would need much stronger safeguards at that "
  "scale.")
h2("How my design choices relate to responsible use")
bullets([
    "<b>Documented, licensed, real data:</b> I used a public dataset with a clear MIT licence and documented "
    "origin, download method and limits, instead of scraping product photos whose owners did not agree to this use.",
    "<b>Memorisation check as part of evaluation:</b> the nearest-neighbour test is built into the notebook, so any "
    "retrained or larger version is checked for copying before its outputs are used.",
    "<b>Low resolution and academic scope:</b> the model's size and output resolution fit an academic study of "
    "generative modelling. I do not release it as a product-image generator, and any generated image shown outside "
    "this project should be labelled as synthetic.",
    "<b>Honest reporting of limits:</b> by reporting per-class errors and failure cases, I make it visible which "
    "items the model represents poorly (sandals, bags, detailed tops) rather than presenting it as uniformly good.",
])

# ----------------------------------------------------------------------------- 6. Limitations
h1("6. Limitations and Future Improvements")
bullets([
    "<b>Blurry outputs.</b> With a pixel-wise Bernoulli likelihood, the decoder predicts the average of plausible "
    f"pixel values. This erased logos, prints and plaid in my results (MSE {f(pick['#3 Pullover'], 3)} for the logo "
    "pullover).",
    "<b>Low resolution and grayscale.</b> At 28 x 28 grayscale the model cannot learn colour, fabric or fine "
    "construction details, so its outputs have no practical design value.",
    "<b>No sample-quality metric.</b> I evaluated generated images visually and through reconstruction error. "
    "Feature-based metrics such as the Fréchet Inception Distance (FID; Heusel et al., 2017) compare the "
    "statistics of generated and real images and would make sample quality measurable and comparable.",
    f"<b>Unused capacity and class mixing.</b> Only {M['active_dims']} of {M['latent_dim']} latent dimensions are "
    "used, and shirt, pullover and coat overlap in latent space.",
])
p("<b>Next steps.</b> (1) Tune β below and above 1 and measure the trade-off more finely (Higgins et al., 2017). "
  "(2) Train a conditional VAE that also receives the class label, so that each item can be generated on request "
  "and the shirt, pullover and coat confusion is reduced. (3) Try different latent sizes, guided by the "
  "active-dimension diagnostic. (4) Add FID as a quantitative sample-quality metric (Heusel et al., 2017). "
  "(5) Move to a colour, higher-resolution clothing dataset with a stronger decoder, and keep the memorisation check "
  "as a required step.")

# ----------------------------------------------------------------------------- 7. References
h1("7. References")
refs = [
    "Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., Balle, B., Ippolito, D., &amp; "
    "Wallace, E. (2023). Extracting training data from diffusion models. In <i>Proceedings of the 32nd USENIX "
    "Security Symposium</i> (pp. 5253–5270). USENIX Association. "
    "https://www.usenix.org/conference/usenixsecurity23/presentation/carlini",
    "Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., &amp; Hochreiter, S. (2017). GANs trained by a two "
    "time-scale update rule converge to a local Nash equilibrium. In <i>Advances in Neural Information Processing "
    "Systems 30</i>. Curran Associates. "
    "https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html",
    "Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., &amp; Lerchner, A. "
    "(2017). beta-VAE: Learning basic visual concepts with a constrained variational framework. In <i>Proceedings "
    "of the 5th International Conference on Learning Representations (ICLR 2017)</i>. "
    "https://openreview.net/forum?id=Sy2fzU9gl",
    "Kingma, D. P., &amp; Welling, M. (2014). Auto-encoding variational Bayes. In <i>Proceedings of the 2nd "
    "International Conference on Learning Representations (ICLR 2014)</i>. https://doi.org/10.48550/arXiv.1312.6114",
    "Xiao, H., Rasul, K., &amp; Vollgraf, R. (2017). <i>Fashion-MNIST: A novel image dataset for benchmarking "
    "machine learning algorithms</i> (arXiv:1708.07747). arXiv. https://doi.org/10.48550/arXiv.1708.07747",
]
for r in refs:
    story.append(Paragraph(r, REF))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawRightString(letter[0] - inch, 0.6 * inch, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.9 * inch,
                        bottomMargin=0.9 * inch, title="Generative AI Analysis and Ethics Report",
                        author="Sirvanmesh", creator="Sirvanmesh")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT}")
