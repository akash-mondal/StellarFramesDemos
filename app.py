import gradio as gr
import numpy as np
import torch
import os
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import requests
current_directory = os.path.abspath(os.getcwd())
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
response = requests.get(url)
if response.status_code == 200:
    file_content = response.content
    filename = url.split("/")[-1]
    with open(filename, "wb") as file:
        file.write(file_content)
    print(f"File '{filename}' downloaded and saved in the current directory.")
else:
    print("Failed to download the file.")
device="cuda"
sam_checkpoint="sam_vit_h_4b8939.pth"
model_type="vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor=SamPredictor(sam)
pipe=StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe=pipe.to(device)
selected_pixels=[]
a = ""
b = ""
bg =""
tg =""
an =""
tag =""
clor =""
clor1 = 0
clor2 = 0
clor3 = 0
def ProductName(name):
    global a
    if not name:
        return "❌❌❌❌❌❌"
    a = name
    return "✅✅✅✅✅✅"
def ProductDes(dsc):
    global b
    if not dsc:
        return "❌❌❌❌❌❌"
    b = dsc
    return "✅✅✅✅✅✅"
def PaLM():
    import google.generativeai as palm
    import re
    palm.configure(api_key="AIzaSyD4dKX1oGETGVATNENZ2Ih0TAEBKWQ4qXU")

    defaults = {
      'model': 'models/text-bison-001',
      'temperature': 1,
      'candidate_count': 8,
      'top_k': 40,
      'top_p': 0.95,
      'max_output_tokens': 1024,
      'stop_sequences': [],
      'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":2},{"category":"HARM_CATEGORY_TOXICITY","threshold":4},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":4},{"category":"HARM_CATEGORY_DANGEROUS","threshold":4}],
    }
    global a
    global b
    global bg
    global tg
    global an
    global clor
    global clor1
    global clor2
    global clor3
    INPUT = "Product:" + a + "/nProduct Description:"+ b
    prompt = f"""Target Niche Audience
    INPUT: Product : Pen
    Product Description : Discover a pen that transcends writing – it embodies precision, elegance, and comfort. With an ultra-smooth nib, sleek metal design, and ergonomic grip, every stroke is a masterpiece. This refillable, hand-assembled instrument makes for a timeless gift, elevating your writing to new heights of sophistication and excellence.

    Audience: Pen Enthusiasts
    Prompt: An antique mahogany writing desk bathed in soft, golden lamplight
    Targeted Keywords: Age: 40-70
    Gender: Both
    Keywords for Targeted Advertisements: Collectible pens, fountain pens, vintage elegance, craftsmanship
    Tagline: Elevate Elegance
    TaglineColor: 100, 100, 100
    INPUT: Product : Cutting-Edge Smartwatch
    Product Description: Stay ahead of the tech curve with our cutting-edge smartwatch. Packed with innovative features, it seamlessly integrates with your digital life, keeping you connected and informed
    Audience: Tech Enthusiasts
    Prompt: A futuristic tech lab filled with holographic displays and neon accents.
    Targeted Keywords: Age: 20-45
    Gender: Both, Keywords for Targeted Advertisements: Innovative tech, advanced gadgets, futuristic design, cutting-edge technology
    Tagline: Future on Your Wrist
    TaglineColor: 51, 204, 51
    INPUT: Product : Pen
    Product Description : Discover a pen that transcends writing – it embodies precision, elegance, and comfort. With an ultra-smooth nib, sleek metal design, and ergonomic grip, every stroke is a masterpiece. This refillable, hand-assembled instrument makes for a timeless gift, elevating your writing to new heights of sophistication and excellence.
    Audience: Creative Professionals
    Prompt: A modern, minimalist studio with a view of a city skyline at night.
    Targeted Keywords: Age: 25-50
    Gender: Both, Keywords for Targeted Advertisements: Designer pens, creative tools, professional writing, sleek design
    Tagline: Design. Write. Inspire.
    TaglineColor: 0, 102, 153
    INPUT: Product : Pen
    Product Description : Discover a pen that transcends writing – it embodies precision, elegance, and comfort. With an ultra-smooth nib, sleek metal design, and ergonomic grip, every stroke is a masterpiece. This refillable, hand-assembled instrument makes for a timeless gift, elevating your writing to new heights of sophistication and excellence.
    Audience: Students and Academics


    Prompt: A cozy, well-lit library filled with rows of ancient books
    Targeted Keywords: Age: 18-35,
    Gender: Both, Keywords for Targeted Advertisements: Study aids, academic excellence, ergonomic pens, comfortable writing
    Tagline: Scholar's Precision Tool
    TaglineColor: 51, 51, 51
    INPUT: Product: Pen,
    Product Description:
    Discover a pen that transcends writing – it embodies precision, elegance, and comfort. With an ultra-smooth nib, sleek metal design, and ergonomic grip, every stroke is a masterpiece. This refillable, hand-assembled instrument makes for a timeless gift, elevating your writing to new heights of sophistication and excellence.

    Audience: Luxury Gift Shoppers
    Prompt: A high-end boutique with soft, diffused lighting showcasing elegant products.
    Targeted Keywords: Age: 30-60
    Gender: Both, Keywords for Targeted Advertisements: Luxury gifts, refined stationery, timeless elegance, sophisticated presents
    Tagline: Elegance Redefined
    TaglineColor: 102, 0, 102
    INPUT: Product : Cutting-Edge Smartwatch
    Product Description: Stay ahead of the tech curve with our cutting-edge smartwatch. Packed with innovative features, it seamlessly integrates with your digital life, keeping you connected and informed
    Audience: Business Professionals
    Prompt: A sophisticated, high-rise office with a glass wall showcasing a bustling city.
    Targeted Keywords: Age: 25-55
    Gender: Both
    Keywords for Targeted Advertisements: Business productivity, smart workwear, connectivity on-the-go, professional edge
    Tagline: Elevate Your Efficiency
    TaglineColor: 0, 102, 204
    INPUT: Product: Mechanical Keyboard
    Product Description : Tactile , Modern , Black , Wired
    Audience: Gaming Enthusiasts
    Prompt: A high-tech gaming setup with neon accents, immersive displays, and a panoramic view of a virtual battlefield.
    Targeted Keywords: Age: 15-35,
    Gender: Both, Keywords for Targeted Advertisements: Gaming gear, competitive edge, responsive keys, esports
    Tagline: Game with Precision
    TaglineColor: 0, 153, 204
    INPUT: Product: Mechanical Keyboard
    Product Description : Tactile , Modern , Black , Wired
    Audience: Professional Writers
    Prompt: A cozy, book-filled study with warm, classic wooden furniture and antique typewriters
    Targeted Keywords: Age: 25-60
    Gender: Both
    Keywords for Targeted Advertisements: Writing tools, tactile keys, timeless design, productivity
    Tagline: Write with Elegance
    TaglineColor: 102, 51, 0
    INPUT: Product: Premium Espresso Machine
    Product Description: Elevate your mornings with our premium espresso machine. Craft barista-quality coffee at home with a sleek, stainless steel design and intuitive controls.
    Audience: Coffee Enthusiasts
    Prompt: A cozy coffee nook with rustic decor, filled with the rich aroma of freshly brewed coffee.
    Targeted Keywords: Age: 30-60
    Gender:Both
    Keywords for Targeted Advertisements:
    Espresso machine
    Home coffee brewing
    Barista-quality coffee
    Coffee aficionado
    Stainless steel appliances
    Coffee machine reviews
    Morning coffee ritual
    Coffee equipment
    Coffee lovers
    Coffee brewing techniques
    Tagline: Brew Perfection Daily
    TaglineColor: 102, 51, 0
    INPUT: {INPUT}
    Audience:"""

    response = palm.generate_text(
      **defaults,
      prompt=prompt
    )
    print(response.result)
    text=response.result
    pattern = r"^(?P<an>.+?)\n\s*?(?P<bg>Prompt: .+?)\n\s*?(?P<tg>Targeted Keywords: .+?)\n\s*?(?P<tag>Tagline: .+?)\n\s*?(?P<clor>TaglineColor: (\d+), (\d+), (\d+))$"

    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)

    if match:
        an = match.group("an")
        bg = match.group("bg")
        tg = match.group("tg")
        tag = match.group("tag").strip()
        clor = match.group("clor")
        clor1 = int(match.group(6))
        clor2 = int(match.group(7))
        clor3 = int(match.group(8))
        bg = bg + "Nikon D850. High quality product photograph.8k,Canon50,Dramatic cinematic lighting,wlop and ross tran"
    else:
        return("NOT VERIFIED PLEASE TRY AGAIN")
    return("VERIFIED")
print(tag)
def cart1():
  global tg
  return(tg)
def cart2():
  global an
  return(an)
def cart4():
  global clor
  return(clor)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
     """
    # **StellarFrame DEMO!**

    Thanks for testing my app , if you encounter any bugs or have any suggestions , contact through github repo - https://github.com/akash-mondal/stellarframe
    """
    )
    with gr.Row():
        gr.Markdown(
            """
            # **STEP 1 - Sumbit the details of the product you have selected**
            """
        )
    with gr.Row():
        inp = gr.Textbox(placeholder="Examples - Car/Shoe/Headphones/Mouse/Tshirt -NO BRAND NAME HERE ",label="Enter Object/Product Name",show_label=True,scale=9)
        out = gr.Textbox(placeholder="NO DATA YET",label="Confirmation Box",show_label=True,scale=1)
    with gr.Row():
      inp1 = gr.Textbox(placeholder="Enter the Features of the Product Example- Stylish , Metallic , Modern , Classic etc",max_lines=1,label="Product Description",show_label=True,scale=9)
      out1 = gr.Textbox(placeholder="NO DATA YET",label="Confirmation Box",show_label=True,scale=1)
    with gr.Row():
       btnn = gr.Button("Submit Details")
       btnn.click(fn=ProductDes, inputs=inp1 ,outputs=out1)
       btnn.click(fn=ProductName, inputs=inp ,outputs=out)
    with gr.Row():
       gr.Markdown(
            """
            # **STEP 2 - Verify The Details , Only Proceed After Verification**

            If data is **NOT VERIFIED** then try verifying again or change the Details to be shorter and more precise and then them again and try verifying(example input-https://app.gemoo.com/share/image-annotation/558310733383217152?codeId=DGqWxkWQoA47O&origin=imageurlgenerator)
            """
       )
    with gr.Row():
      uio = gr.Textbox(label="Critical Confrimation")
    bt2 = gr.Button("AFTER SUBMITING THE DETAILS CLICK HERE - AND WAIT TILL IT SHOWS VERIFIED ABOVE")
    bt2.click(fn=PaLM,outputs=uio)
    with gr.Row():
        gr.Markdown(
            """
            # **STEP 3 - Upload your Product Image in the Block Below**

            # IMAGE SHOULD HAVE SQUARE DIMENSIONS (SAME HEIGHT AND WIDTH) KINDLY RESIZE YOUR IMAGE HERE IF YOU WISH TO - https://www.birme.net/
            IMAGE SHOULD HAVE THE PRODUCT IN THE CENTER AND SHOULD BE EASY TO PICK OUT (BACKGROUND SHOULD BE PLAIN FOR BEST RESULTS)

            """
        )
    with gr.Row():
        input_img = gr.Image(label="Input")
    with gr.Row():
        gr.Markdown(
            """
            # **STEP 4 - AFTER UPLOADING THE PRODUCT IMAGE ABOVE , CLICK ON THE PRODUCT , CONTINUE TILL THE IMAGE TAB ON THE SIDE SHOWS YOUR PRODUCT IN BLACK AND THE BACKGROUND AS WHITE**

            **Example-** https://s11.gifyu.com/images/SgPT4.gif

            MAKE SURE THE PRODUCT IS COMPLETELY COVERED IN BLACK AND THE BACKGROUND IS ALL WHITE

            """
        )
        mask_img = gr.Image(label="Mask",interactive=False)
    with gr.Row():
        gr.Markdown(
            """
            # **STEP 5 - Click on Sumbit and let the model generate stunning ad artwork , use the Target Keywords to Target these ads to the right demographic on Online Advertising Platforms such as Google ads and Meta ads (facebook)**

            You Might Not Get The Best Image On The First Try So Please Keep Submiting Till You Are Satisfied With The Results , Then Move to The Next Step
            """
        )
    with gr.Row():
        output_img1=gr.Image(label="Output",interactive=False)
        tagbox2=gr.Textbox(label="Demographic")
        tagbox1=gr.Textbox(label="TARGETED KEYWORDS")
        tagbox4=gr.Textbox(label="RGB VALUE OF THE SUGGESTED COLOR FOR THE CAMPAIGN")
        submit1=gr.Button("Submit")
    with gr.Row():
        gr.Markdown(
            """
            # **STEP 6 - Click on New Target Audience to generate brand new Ad Artwork targeting a Different Demographic , Click on Submit Button in Step 5 (which is above) to generate said new artwork after getting the confirmations below**

            You can start again from step 1 to change the product and upload a new image
            """
        )
    with gr.Row():
        btw=gr.Button("New Target Audience")
        tagbox5=gr.Textbox(label="Confirmation")
    def generate_mask(image , evt: gr.SelectData):
        selected_pixels.append(evt.index)
        predictor.set_image(image)
        input_points=np.array(selected_pixels)
        input_labels=np.ones(input_points.shape[0])
        mask, _, _ =predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        mask=np.logical_not(mask)
        mask=Image.fromarray(mask[0, :, :])
        return mask
    def inpaint(image,mask):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image=image.resize((512,512))
        mask=mask.resize((512,512))
        output=pipe(prompt=bg, image=image , mask_image=mask,).images[0]
        return output
    input_img.select(generate_mask,[input_img],[mask_img])
    submit1.click(inpaint,inputs=[input_img ,mask_img],outputs=[output_img1],)
    submit1.click(fn=cart1,outputs=tagbox1)
    submit1.click(fn=cart2,outputs=tagbox2)
    submit1.click(fn=cart4,outputs=tagbox4)
    btw.click(fn=PaLM,outputs=tagbox5)
if __name__== "__main__":
    demo.launch(share=True,debug=True)