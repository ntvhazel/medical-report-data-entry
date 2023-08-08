import contextlib
from io import BytesIO
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
from PIL import Image
import json
import pandas as pd
import imutils
from skimage.filters import threshold_local
import cv2
import os
from ocr import inference, merge
import mysql.connector
from streamlit_js_eval import streamlit_js_eval
import time
from PIL import ImageOps



st.set_page_config(
    page_title="Nhập liệu bệnh án",
    page_icon="🖼️",
    layout="wide",
)




# ---------- HEADER ----------
st.title("Nhập liệu bệnh án")


# ---------- FUNCTIONS ----------
def _reset(key: str) -> None:
    if key == "all":
        st.session_state["rotate_slider"] = 0
        st.session_state["brightness_slider"] = st.session_state[
            "saturation_slider"
        ] = st.session_state["contrast_slider"] = 100
        st.session_state["bg"] = st.session_state["gray_bw"] = 0
    elif key == "rotate_slider":
        st.session_state["rotate_slider"] = 0
    elif key == "checkboxes":
        st.session_state[
            "gray_bw"
        ] = 0
    else:
        st.session_state[key] = 100
        
def scroll_to_top():
    """Scrolls to the top of the page."""
    js = '''
    <script>
        var body = window.parent.document.querySelector(".main");
        console.log(body);
        body.scrollTop = 0;
    </script>
    '''
    if "scroll_to_top" not in st.session_state:
        st.session_state["scroll_to_top"] = js
    st.components.v1.html(js)
    st.write(js, unsafe_allow_html=True)

def order_points(pts):
    # initializing the list of coordinates to be ordered
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    # top-left point will have the smallest sum
    rect[0] = pts[np.argmin(s)]
    # bottom-right point will have the largest sum
    rect[2] = pts[np.argmax(s)]

    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # returns ordered coordinates
    return rect


def perspective_transform(image, pts):
    # unpack the ordered coordinates individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    # Apply the transform matrix
    warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

    # return the warped image
    return warped

ready = False
# ---------- OPERATIONS ----------

option = st.radio(
    label="Upload an image, take one with your camera, or load image from a URL",
    options=(
        "Upload an image ⬆️",
        "Take a photo with my camera 📷",
        "Load image from a URL 🌐",
    ),
    help="Uploaded images are deleted from the server when you\n* upload another image\n* clear the file uploader\n* close the browser tab",
)

if option == "Take a photo with my camera 📷":
    upload_img = st.camera_input(
        label="Take a picture",
    )
    mode = "camera"

elif option == "Upload an image ⬆️":
    upload_img = st.file_uploader(
        label="Upload an image",
        type=["bmp", "jpg", "jpeg", "png", "svg"]
    )
    mode = "upload"

elif option == "Load image from a URL 🌐":
    url = st.text_input(
        "Image URL",
        key="url",
    )
    mode = "url"

    if url != "":
        try:
            response = requests.get(url)
            upload_img = Image.open(BytesIO(response.content))
        except:
            st.error("The URL does not seem to be valid.")

with contextlib.suppress(NameError):
    if upload_img is not None:
        pil_img = (
            upload_img.convert("RGB")
            if mode == "url"
            else Image.open(upload_img).convert("RGB")
        )
        img_arr = np.asarray(pil_img)





        with st.container():
            lcol, rcol = st.columns(2)
            image = Image.fromarray(img_arr)
            image = ImageOps.exif_transpose(image)
            right = 100
            left = 100
            top = 100
            bottom = 100
            width, height = image.size

            new_width = width + right + left
            new_height = height + top + bottom
                
            result = Image.new(image.mode, (new_width, new_height), (0,0,0))
            result.paste(image, (left, top))
            result = np.asarray(result)
            ratio = result.shape[0] / 500.0
            img_resize = imutils.resize(result, height=500)
            copy = result.copy()
            gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            edged_img = cv2.Canny(blurred_image, 75, 200)
            cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    doc = approx
                    break
                    
            p = []
            for d in approx:
                tuple_point = tuple(d[0])
                cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)
                p.append(tuple_point)
            warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
            warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
                # image = warped_image
            image = Image.fromarray(warped_image)
        flag = True


        expander = st.expander("Advanced")
        # ---------- OTHER OPERATIONS ----------
        # ---------- 1ST ROW ----------
        with expander.container():
            lcol, mcol, rcol = st.columns(3)

            # ---------- ROTATE ----------
            if "rotate_slider" not in st.session_state:
                st.session_state["rotate_slider"] = 0
            degrees = lcol.slider(
                "Drag slider to rotate image clockwise 🔁",
                min_value=0,
                max_value=360,
                value=st.session_state["rotate_slider"],
                key="rotate_slider",
            )
            rotated_img = image.rotate(360 - degrees)
            lcol.image(
                rotated_img,
                use_column_width="auto",
                caption=f"Rotated by {degrees} degrees clockwise",
            )
            if lcol.button(
                "↩️ Reset Rotation",
                on_click=_reset,
                use_container_width=True,
                kwargs={"key": "rotate_slider"},
            ):
                lcol.success("Rotation reset to original!")

            if flag:
                # ---------- BRIGHTNESS ----------
                if "brightness_slider" not in st.session_state:
                    st.session_state["brightness_slider"] = 100
                brightness_factor = mcol.slider(
                    "Drag slider to change brightness 💡",
                    min_value=0,
                    max_value=200,
                    value=st.session_state["brightness_slider"],
                    key="brightness_slider",
                )
                brightness_img = np.asarray(
                    ImageEnhance.Brightness(rotated_img).enhance(
                        brightness_factor / 100
                    )
                )
                mcol.image(
                    brightness_img,
                    use_column_width="auto",
                    caption=f"Brightness: {brightness_factor}%",
                )
                if mcol.button(
                    "↩️ Reset Brightness",
                    on_click=_reset,
                    use_container_width=True,
                    kwargs={"key": "brightness_slider"},
                ):
                    mcol.success("Brightness reset to original!")

                # ---------- SATURATION ----------
                if "saturation_slider" not in st.session_state:
                    st.session_state["saturation_slider"] = 100
                saturation_factor = rcol.slider(
                    "Drag slider to change saturation",
                    min_value=0,
                    max_value=200,
                    value=st.session_state["saturation_slider"],
                    key="saturation_slider",
                )
                saturation_img = np.asarray(
                    ImageEnhance.Color(Image.fromarray(brightness_img)).enhance(
                        saturation_factor / 100
                    )
                )
                rcol.image(
                    saturation_img,
                    use_column_width="auto",
                    caption=f"Saturation: {saturation_factor}%",
                )
                if rcol.button(
                    "↩️ Reset Saturation",
                    on_click=_reset,
                    use_container_width=True,
                    kwargs={"key": "saturation_slider"},
                ):
                    rcol.success("Saturation reset to original!")

                st.markdown("""---""")

                # ---------- 2ND ROW ----------
                with st.container():
                    lcol, mcol, rcol = st.columns(3)
                    # ---------- CONTRAST ----------
                    if "contrast_slider" not in st.session_state:
                        st.session_state["contrast_slider"] = 100
                    contrast_factor = lcol.slider(
                        "Drag slider to change contrast",
                        min_value=0,
                        max_value=200,
                        value=st.session_state["contrast_slider"],
                        key="contrast_slider",
                    )
                    contrast_img = np.asarray(
                        ImageEnhance.Contrast(Image.fromarray(saturation_img)).enhance(
                            contrast_factor / 100
                        )
                    )
                    lcol.image(
                        contrast_img,
                        use_column_width="auto",
                        caption=f"Contrast: {contrast_factor}%",
                    )
                    if lcol.button(
                        "↩️ Reset Contrast",
                        on_click=_reset,
                        use_container_width=True,
                        kwargs={"key": "contrast_slider"},
                    ):
                        lcol.success("Contrast reset to original!")

                    # ---------- SHARPNESS ----------
                    if "sharpness_slider" not in st.session_state:
                        st.session_state["sharpness_slider"] = 100
                    sharpness_factor = mcol.slider(
                        "Drag slider to change sharpness",
                        min_value=0,
                        max_value=200,
                        value=st.session_state["sharpness_slider"],
                        key="sharpness_slider",
                    )
                    sharpness_img = np.asarray(
                        ImageEnhance.Sharpness(Image.fromarray(contrast_img)).enhance(
                            sharpness_factor / 100
                        )
                    )
                    mcol.image(
                        sharpness_img,
                        use_column_width="auto",
                        caption=f"Sharpness: {sharpness_factor}%",
                    )
                    if mcol.button(
                        "↩️ Reset Sharpness",
                        on_click=_reset,
                        use_container_width=True,
                        kwargs={"key": "sharpness_slider"},
                    ):
                        mcol.success("Sharpness reset to original!")


        # ---------- FINAL OPERATIONS ----------
        st.subheader("View Results")
        lcol, rcol = st.columns(2)
        lcol.image(
            img_arr,
            use_column_width="auto",
            caption=f"Original Image ({pil_img.size[0]} x {pil_img.size[1]})",
        )

        try:
            final_image = sharpness_img
        except NameError:
            final_image = rotated_img

        rcol.image(
            final_image,
            use_column_width="auto",
            caption=f"Final Image ({final_image.shape[1]} x {final_image.shape[0]})"
            if flag
            else f"Final Image ({final_image.size[1]} x {final_image.size[0]})",
        )

        if flag:
            Image.fromarray(final_image).save("final_image.png")
            ready = True
        else:
            final_image.save("final_image.png")
            ready = True

        if st.button(
            "↩️ Reset All",
            on_click=_reset,
            use_container_width=True,
            kwargs={"key": "all"},
        ):
            st.success(body="Image reset to original!", icon="↩️")
if ready:
    
        result = inference("final_image.png")
        for keys in result.keys:
            if result[keys] == "None" or result[keys] == "Null":
                del result[keys]
        df = pd.DataFrame.from_dict(result, orient='index', columns=['Value'])
        # df = df.transpose()
        # st.write(df)
        edited_df = st.data_editor(df, key="data_editor", use_container_width =True)
        # edited_df = edited_df.transpose()
        final_result = pd.DataFrame.to_dict(edited_df)
        final_result = final_result['Value']
        # st.write(final_result)
        # if st.button("Lưu"):
        #     columns = ', '.join("`" + str(x).replace(':', '') + "`" for x in final_result.keys())
        #     values = ', '.join("'" + str(x) + "'" for x in final_result.values())
            
        #     sql = "INSERT INTO %s ( %s ) VALUES ( %s );" % ('med_records', columns, values)
        #     st.write(sql)
        #     # try:
        #     cursor = mydb.cursor()
        #     cursor.execute(sql)
        #     mydb.commit()
        #     st.success("Hoàn thành!")
        #     time.sleep(3)
        #     streamlit_js_eval(js_expressions="parent.window.location.reload()")
            # except mysql.connector.Error as error:
            #     st.error("Có lỗi xảy ra!")


                