from pathlib import Path
import streamlit as st


def show_header():
    with st.container():
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            col_img, col_text = st.columns([1, 5])
            with col_img:
                st.image("book.jpg", width=60)
            with col_text:
                st.markdown("<h1 style='margin-top: 10px; color: #cc3333'>XỬ LÝ ẢNH SỐ (DIP)</h2>", unsafe_allow_html=True)

def show_authors():
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Giáo viên hướng dẫn: ThS Trần Tiến Đức</p>
            <p>Nhóm thực hiện:</p>
            <p>Nguyễn Chí Thanh - 22110226</p>
            <p>Trần Như Quỳnh - 22110218</p>
        </div>
        """,
        unsafe_allow_html=True
    )
