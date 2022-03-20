import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";

interface ScatterImagesProps {
  intensities: any;
}

const ScatterImage = (props: ScatterImagesProps) => {
  return (
      <TransformWrapper>
        <TransformComponent>
          {/* <img
        style={{maxWidth: "200%"}}
        src="https://www.pexels.com/photo/2246476/download/?search_query=4k%20wallpaper&tracking_id=erqwiyvsx8"
      /> */}
          <img
            className="introImg"
            alt=""
            style={{ maxWidth: "100%" }}
            src={`data:image/jpeg;base64,${props.intensities}`}
          />
        </TransformComponent>
      </TransformWrapper>
  );
};

export default ScatterImage;
