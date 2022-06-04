import { jsx as _jsx } from "react/jsx-runtime";
var ScatterImage = function (props) {
    return (
    // <img
    //   style={{ maxWidth: "100%" }}
    //   src="https://www.pexels.com/photo/2246476/download/?search_query=4k%20wallpaper&tracking_id=erqwiyvsx8"/>
    // <canvas style={{ background:"red", width: "1475", height: "1679" }} id="canvas" ref={canvasRef} />
    _jsx("img", { alt: "", style: { height: "100%", width: "100%", display: 'block' }, src: "data:image/jpeg;base64,".concat(props.intensities) }));
};
export default ScatterImage;
