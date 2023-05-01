import React from "react";

interface JsonCallback {
    (value: any, key: any): void;
  }

const useJsonCallback = (): [{}, JsonCallback]  => {
    const [json, setJson] = React.useState({});
    const jsonCallback: JsonCallback = (value :any, key: any) => {
        json[key] = value;
        console.log(json)
        setJson({ ...json });
    };

    return [json, jsonCallback]
}

export default useJsonCallback