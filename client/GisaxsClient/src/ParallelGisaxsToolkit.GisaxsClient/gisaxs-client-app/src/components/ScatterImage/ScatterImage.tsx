interface ScatterImagesProps {
  intensities: any;
  width: any;
  height: any;
}

const ScatterImage = (props: ScatterImagesProps) => {
  return (
    <img
      alt=""
      style={{ height: "100%", width: "100%", display: 'block' }}
      src={props.intensities}
    />
  );
};

export default ScatterImage;
