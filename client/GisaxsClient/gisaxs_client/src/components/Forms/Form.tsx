import JSONForm from "@rjsf/material-ui";
import type { JSONSchema7 } from "json-schema";
import type { UiSchema } from "@rjsf/core";
import CustomObjectFieldTemplate from "./CustomObjectFieldTemplate";
import CustomObjectFieldTemplateSmall from "./CustomObjectFieldTemplateSmall";
import CustomObjectFieldTemplateCollapse from "./CustomObjectFieldTemplateCollapse";
import CustomObjectFieldTemplateLarge from "./CustomObjectFieldTemplateLarge";

import configSchema from "./shapes.schema.json";

const uiSchema: UiSchema = {
  shapes: {
    items: {
      "ui:ObjectFieldTemplate": CustomObjectFieldTemplateCollapse,

      radius: {
        "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
        mean: {
          "ui:widget": "range",
        },
        stddev: {
          "ui:widget": "range",
        },
      },
      height: {
        "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
        mean: {
          "ui:widget": "range",
        },
        stddev: {
          "ui:widget": "range",
        },
      },
      refraction: {
        "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
      },
      locations: {
        items: {
          "ui:ObjectFieldTemplate": CustomObjectFieldTemplateSmall,
          x: {
            "ui:widget": "range",
          },
          y: {
            "ui:widget": "range",
          },
          z: {
            "ui:widget": "range",
          },
        },
      },
    },
  },
};

const validConfigSchema: any = configSchema;

function Form(props: { callback: Function; formData: any }) {
  const changeHandler = (value: any) => {
    props.callback(value.formData);
  };

  return (
    <JSONForm
      children={true}
      schema={validConfigSchema}
      formData={props.formData}
      uiSchema={uiSchema}
      onChange={(value) => {
        changeHandler(value);
      }}
    />
  );
}

export default Form;
