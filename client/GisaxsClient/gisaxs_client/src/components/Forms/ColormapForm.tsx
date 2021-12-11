import JSONForm from "@rjsf/material-ui";
import type { JSONSchema7 } from "json-schema";
import type { UiSchema } from "@rjsf/core";
import CustomObjectFieldTemplate from "./CustomObjectFieldTemplate";
import CustomObjectFieldTemplateSmall from "./CustomObjectFieldTemplateSmall";

const uiSchema: UiSchema = {
  colormap: {},
};

const schema2: JSONSchema7 = {
  type: "object",
  properties: {
    colormapName: {
      $ref: "#/definitions/colormapName",
    },
  },
  definitions: {
    colormapName: {
      type: "string",
      oneOf: [
        {
          type: "string",
          title: "twilightShifted",
          enum: ["twilightShifted"],
        },
        {
          type: "string",
          title: "twilight",
          enum: ["twilight"],
        },
        {
          type: "string",
          title: "autumn",
          enum: ["autumn"],
        },
        {
          type: "string",
          title: "parula",
          enum: ["parula"],
        },
        {
          type: "string",
          title: "bone",
          enum: ["bone"],
        },
        {
          type: "string",
          title: "cividis",
          enum: ["cividis"],
        },
        {
          type: "string",
          title: "cool",
          enum: ["cool"],
        },
        {
          type: "string",
          title: "hot",
          enum: ["hot"],
        },
        {
          type: "string",
          title: "hsv",
          enum: ["hsv"],
        },
        {
          type: "string",
          title: "inferno",
          enum: ["inferno"],
        },
        {
          type: "string",
          title: "jet",
          enum: ["jet"],
        },
        {
          type: "string",
          title: "magma",
          enum: ["magma"],
        },
        {
          type: "string",
          title: "ocean",
          enum: ["ocean"],
        },
        {
          type: "string",
          title: "pink",
          enum: ["pink"],
        },
        {
          type: "string",
          title: "plasma",
          enum: ["plasma"],
        },
        {
          type: "string",
          title: "rainbow",
          enum: ["rainbow"],
        },
        {
          type: "string",
          title: "spring",
          enum: ["spring"],
        },
        {
          type: "string",
          title: "summer",
          enum: ["summer"],
        },
        {
          type: "string",
          title: "viridis",
          enum: ["viridis"],
        },
        {
          type: "string",
          title: "winter",
          enum: ["winter"],
        },
      ],
      default: "twilightShifted",
    },
  },
};

function ColormapForm(props: { callback: Function; formData: any }) {
  const changeHandler = (value: any) => {
    props.callback(value.formData);
  };

  return (
    <JSONForm
      children={true}
      schema={schema2}
      formData={props.formData}
      uiSchema={uiSchema}
      onChange={(value) => {
        changeHandler(value);
      }}
    />
  );
}

export default ColormapForm;
