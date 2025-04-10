from UnumCloud.unum_main import unum_main
from NlpConnect.nlp_main import nlp_connect_main
from Salesforce.salesforce_main import salesforce_main
from Llama.llama_main import llama_main
from Florence.florence_main import florence_main
from Google.gemma_main import gemma_main
from Qwen.qwen_main import qwen_main
from Llava.llava_mian import llava_main
from Bipin.bipin_main import bipin_main
from Noamrot.noamrot_main import noamrot_main

image_links = [
    "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1599566150163-29194dcaad36?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1521566652839-697aa473761a?q=80&w=1742&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1552058544-f2b08422138a?q=80&w=1899&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1554151228-14d9def656e4?q=80&w=1886&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1504593811423-6dd665756598?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
]

if __name__ == "__main__":
    #salesforce_main(image_links)
    #llama_main(image_links)
    #nlp_connect_main(image_links)
    # unum_main(image_links)
    #florence_main(image_links)
    ##gemma_main(image_links)
    # qwen_main(image_links)
    llava_main(image_links)
    # bipin_main(image_links)
    # noamrot_main(image_links)